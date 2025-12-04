import os
import hashlib
import shutil
import time
import uuid
import logging
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Union, Any
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum

from sqlalchemy import create_engine, desc, Column, Integer, String, Float, DateTime, ForeignKey, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship

from modules import shared

Base = declarative_base()


class SearchMode(Enum):
    ONLINE = "online"
    LOCAL = "local"


class ImageRetrievalError(Exception):
    pass


class ValidationError(ImageRetrievalError):
    pass


class ModelLoadError(ImageRetrievalError):
    pass


class CrawlerError(ImageRetrievalError):
    pass


class DatabaseError(ImageRetrievalError):
    pass


@dataclass
class SearchConfig:
    min_similarity_threshold: float = 0.0
    max_results: int = 100
    batch_size: int = 8
    min_image_size: Tuple[int, int] = (200, 200)
    supported_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    max_crawl_threads: int = 4
    crawl_timeout: int = 30
    
    def __post_init__(self):
        assert 0.0 <= self.min_similarity_threshold <= 1.0, \
            f"Similarity threshold must be between 0 and 1, got {self.min_similarity_threshold}"
        assert self.max_results > 0, \
            f"Max results must be positive, got {self.max_results}"
        assert self.batch_size > 0, \
            f"Batch size must be positive, got {self.batch_size}"
        assert len(self.min_image_size) == 2, \
            f"Min image size must be a tuple of 2 integers, got {self.min_image_size}"
        assert all(s > 0 for s in self.min_image_size), \
            f"Min image size dimensions must be positive, got {self.min_image_size}"
        assert len(self.supported_extensions) > 0, \
            "Supported extensions cannot be empty"
        assert self.max_crawl_threads > 0, \
            f"Max crawl threads must be positive, got {self.max_crawl_threads}"


@dataclass
class SearchResult:
    image_path: str
    similarity: float
    rank: int = 0
    
    def __post_init__(self):
        assert isinstance(self.image_path, str), \
            f"Image path must be a string, got {type(self.image_path)}"
        assert len(self.image_path) > 0, \
            "Image path cannot be empty"
        assert isinstance(self.similarity, (int, float)), \
            f"Similarity must be a number, got {type(self.similarity)}"
        assert 0.0 <= self.similarity <= 1.0, \
            f"Similarity must be between 0 and 1, got {self.similarity}"
        assert self.rank >= 0, \
            f"Rank must be non-negative, got {self.rank}"


class Logger:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._logger = logging.getLogger("ImageRetrieval")
        self._logger.setLevel(logging.INFO)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '[%(name)s] %(levelname)s: %(message)s'
            ))
            self._logger.addHandler(handler)
    
    def info(self, message: str) -> None:
        assert isinstance(message, str), "Log message must be a string"
        self._logger.info(message)
    
    def warning(self, message: str) -> None:
        assert isinstance(message, str), "Log message must be a string"
        self._logger.warning(message)
    
    def error(self, message: str) -> None:
        assert isinstance(message, str), "Log message must be a string"
        self._logger.error(message)
    
    def debug(self, message: str) -> None:
        assert isinstance(message, str), "Log message must be a string"
        self._logger.debug(message)


def get_logger() -> Logger:
    return Logger()


def validate_image_path(path: str) -> bool:
    if not isinstance(path, str):
        raise ValidationError(f"Path must be a string, got {type(path)}")
    if len(path) == 0:
        raise ValidationError("Path cannot be empty")
    if not os.path.exists(path):
        raise ValidationError(f"Path does not exist: {path}")
    if not os.path.isfile(path):
        raise ValidationError(f"Path is not a file: {path}")
    return True


def validate_directory_path(path: str) -> bool:
    if not isinstance(path, str):
        raise ValidationError(f"Path must be a string, got {type(path)}")
    if len(path) == 0:
        raise ValidationError("Path cannot be empty")
    if not os.path.exists(path):
        raise ValidationError(f"Directory does not exist: {path}")
    if not os.path.isdir(path):
        raise ValidationError(f"Path is not a directory: {path}")
    return True


def validate_keyword(keyword: str) -> bool:
    if not isinstance(keyword, str):
        raise ValidationError(f"Keyword must be a string, got {type(keyword)}")
    keyword = keyword.strip()
    if len(keyword) == 0:
        raise ValidationError("Keyword cannot be empty")
    if len(keyword) > 256:
        raise ValidationError(f"Keyword too long: {len(keyword)} chars (max 256)")
    return True


def validate_max_num(max_num: int) -> bool:
    if not isinstance(max_num, (int, float)):
        raise ValidationError(f"Max num must be a number, got {type(max_num)}")
    max_num = int(max_num)
    if max_num <= 0:
        raise ValidationError(f"Max num must be positive, got {max_num}")
    if max_num > 100:
        raise ValidationError(f"Max num too large: {max_num} (max 100)")
    return True


def validate_query_id(query_id: Any) -> int:
    if isinstance(query_id, str):
        query_id = query_id.strip()
        if len(query_id) == 0:
            raise ValidationError("Query ID cannot be empty")
        try:
            query_id = int(query_id)
        except ValueError:
            raise ValidationError(f"Query ID must be an integer, got '{query_id}'")
    if not isinstance(query_id, int):
        raise ValidationError(f"Query ID must be an integer, got {type(query_id)}")
    if query_id <= 0:
        raise ValidationError(f"Query ID must be positive, got {query_id}")
    return query_id


def validate_pil_image(image: Any) -> bool:
    if image is None:
        raise ValidationError("Image cannot be None")
    if not isinstance(image, Image.Image):
        raise ValidationError(f"Image must be a PIL Image, got {type(image)}")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValidationError(f"Image dimensions must be positive, got {width}x{height}")
    return True


def validate_embedding(embedding: Any) -> bool:
    if embedding is None:
        raise ValidationError("Embedding cannot be None")
    if not isinstance(embedding, np.ndarray):
        raise ValidationError(f"Embedding must be a numpy array, got {type(embedding)}")
    if embedding.ndim != 1:
        raise ValidationError(f"Embedding must be 1-dimensional, got {embedding.ndim} dimensions")
    if len(embedding) == 0:
        raise ValidationError("Embedding cannot be empty")
    if not np.isfinite(embedding).all():
        raise ValidationError("Embedding contains non-finite values")
    return True


def normalize_path(path: str) -> str:
    assert isinstance(path, str), f"Path must be a string, got {type(path)}"
    path = path.strip()
    path = os.path.normpath(path)
    path = os.path.abspath(path)
    return path


def sanitize_keyword(keyword: str) -> str:
    assert isinstance(keyword, str), f"Keyword must be a string, got {type(keyword)}"
    keyword = keyword.strip()
    keyword = ' '.join(keyword.split())
    return keyword


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = Path(shared.cmd_opts.data_dir if hasattr(shared.cmd_opts, 'data_dir') and shared.cmd_opts.data_dir else ".") / "image_retrieval_data"
TEMP_DIR = DATA_DIR / "temp"
QUERIES_DIR = DATA_DIR / "queries"
DB_PATH = DATA_DIR / "image_retrieval.db"

DEFAULT_CONFIG = SearchConfig()


@dataclass
class CacheConfig:
    max_cache_size_mb: float = 500.0
    max_temp_files: int = 1000
    max_query_images: int = 500
    auto_cleanup_threshold: float = 0.9
    temp_file_max_age_hours: int = 24
    
    def __post_init__(self):
        assert self.max_cache_size_mb > 0, \
            f"Max cache size must be positive, got {self.max_cache_size_mb}"
        assert self.max_temp_files > 0, \
            f"Max temp files must be positive, got {self.max_temp_files}"
        assert self.max_query_images > 0, \
            f"Max query images must be positive, got {self.max_query_images}"
        assert 0.0 < self.auto_cleanup_threshold <= 1.0, \
            f"Auto cleanup threshold must be between 0 and 1, got {self.auto_cleanup_threshold}"
        assert self.temp_file_max_age_hours > 0, \
            f"Temp file max age must be positive, got {self.temp_file_max_age_hours}"


DEFAULT_CACHE_CONFIG = CacheConfig()


class CacheManager:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(CacheManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: CacheConfig = None):
        if self._initialized:
            return
        
        self._logger = get_logger()
        self._config = config or DEFAULT_CACHE_CONFIG
        
        self._temp_dir = TEMP_DIR
        self._queries_dir = QUERIES_DIR
        self._data_dir = DATA_DIR
        
        self._ensure_directories_exist()
        self._initialized = True
        
        self._logger.info("CacheManager initialized")
    
    def _ensure_directories_exist(self) -> None:
        for directory in [self._temp_dir, self._queries_dir, self._data_dir]:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                assert directory.exists(), f"Failed to create directory: {directory}"
            except Exception as e:
                self._logger.error(f"Failed to create directory {directory}: {e}")
    
    def get_directory_size(self, directory: Path) -> int:
        assert isinstance(directory, Path), \
            f"Directory must be a Path, got {type(directory)}"
        
        if not directory.exists():
            return 0
        
        total_size = 0
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    try:
                        total_size += file_path.stat().st_size
                    except (OSError, IOError) as e:
                        self._logger.warning(f"Cannot get size of {file_path}: {e}")
        except Exception as e:
            self._logger.error(f"Error calculating directory size: {e}")
        
        assert total_size >= 0, f"Invalid total size: {total_size}"
        return total_size
    
    def get_file_count(self, directory: Path) -> int:
        assert isinstance(directory, Path), \
            f"Directory must be a Path, got {type(directory)}"
        
        if not directory.exists():
            return 0
        
        count = 0
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    count += 1
        except Exception as e:
            self._logger.error(f"Error counting files: {e}")
        
        assert count >= 0, f"Invalid file count: {count}"
        return count
    
    def get_cache_info(self) -> Dict[str, Any]:
        temp_size = self.get_directory_size(self._temp_dir)
        queries_size = self.get_directory_size(self._queries_dir)
        total_size = self.get_directory_size(self._data_dir)
        
        temp_files = self.get_file_count(self._temp_dir)
        query_files = self.get_file_count(self._queries_dir)
        
        max_size_bytes = self._config.max_cache_size_mb * 1024 * 1024
        usage_ratio = total_size / max_size_bytes if max_size_bytes > 0 else 0
        
        cache_info = {
            'temp_dir': str(self._temp_dir),
            'queries_dir': str(self._queries_dir),
            'data_dir': str(self._data_dir),
            'temp_size_bytes': temp_size,
            'temp_size_mb': temp_size / (1024 * 1024),
            'queries_size_bytes': queries_size,
            'queries_size_mb': queries_size / (1024 * 1024),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'temp_file_count': temp_files,
            'query_file_count': query_files,
            'max_cache_size_mb': self._config.max_cache_size_mb,
            'usage_ratio': usage_ratio,
            'usage_percent': usage_ratio * 100,
            'needs_cleanup': usage_ratio >= self._config.auto_cleanup_threshold
        }
        
        return cache_info
    
    def clear_temp_directory(self) -> int:
        self._logger.info(f"Clearing temp directory: {self._temp_dir}")
        
        cleared_count = 0
        cleared_size = 0
        
        if not self._temp_dir.exists():
            self._logger.info("Temp directory does not exist, nothing to clear")
            return 0
        
        try:
            for file_path in self._temp_dir.iterdir():
                try:
                    if file_path.is_file():
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        cleared_count += 1
                        cleared_size += file_size
                    elif file_path.is_dir():
                        shutil.rmtree(file_path)
                        cleared_count += 1
                except Exception as e:
                    self._logger.warning(f"Failed to delete {file_path}: {e}")
        except Exception as e:
            self._logger.error(f"Error clearing temp directory: {e}")
        
        self._logger.info(
            f"Cleared {cleared_count} items ({cleared_size / (1024 * 1024):.2f} MB)"
        )
        
        return cleared_count
    
    def clear_old_temp_files(self, max_age_hours: int = None) -> int:
        max_age_hours = max_age_hours or self._config.temp_file_max_age_hours
        
        assert isinstance(max_age_hours, int), \
            f"max_age_hours must be an integer, got {type(max_age_hours)}"
        assert max_age_hours > 0, \
            f"max_age_hours must be positive, got {max_age_hours}"
        
        self._logger.info(f"Clearing temp files older than {max_age_hours} hours")
        
        if not self._temp_dir.exists():
            return 0
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        cleared_count = 0
        
        try:
            for file_path in self._temp_dir.rglob('*'):
                if not file_path.is_file():
                    continue
                
                try:
                    file_mtime = file_path.stat().st_mtime
                    file_age = current_time - file_mtime
                    
                    if file_age > max_age_seconds:
                        file_path.unlink()
                        cleared_count += 1
                        self._logger.debug(f"Deleted old file: {file_path}")
                except Exception as e:
                    self._logger.warning(f"Failed to check/delete {file_path}: {e}")
        except Exception as e:
            self._logger.error(f"Error clearing old temp files: {e}")
        
        self._logger.info(f"Cleared {cleared_count} old temp files")
        
        return cleared_count
    
    def clear_old_query_images(self, keep_recent: int = None) -> int:
        keep_recent = keep_recent or self._config.max_query_images
        
        assert isinstance(keep_recent, int), \
            f"keep_recent must be an integer, got {type(keep_recent)}"
        assert keep_recent >= 0, \
            f"keep_recent must be non-negative, got {keep_recent}"
        
        self._logger.info(f"Clearing old query images, keeping {keep_recent} recent")
        
        if not self._queries_dir.exists():
            return 0
        
        try:
            query_files = []
            for file_path in self._queries_dir.iterdir():
                if file_path.is_file():
                    try:
                        mtime = file_path.stat().st_mtime
                        query_files.append((file_path, mtime))
                    except Exception as e:
                        self._logger.warning(f"Cannot get mtime of {file_path}: {e}")
            
            query_files.sort(key=lambda x: x[1], reverse=True)
            
            cleared_count = 0
            for file_path, _ in query_files[keep_recent:]:
                try:
                    file_path.unlink()
                    cleared_count += 1
                except Exception as e:
                    self._logger.warning(f"Failed to delete {file_path}: {e}")
            
            self._logger.info(f"Cleared {cleared_count} old query images")
            
            return cleared_count
        except Exception as e:
            self._logger.error(f"Error clearing old query images: {e}")
            return 0
    
    def enforce_cache_limit(self) -> Dict[str, int]:
        self._logger.info("Enforcing cache limits...")
        
        results = {
            'temp_files_cleared': 0,
            'query_images_cleared': 0,
            'old_files_cleared': 0
        }
        
        cache_info = self.get_cache_info()
        
        if cache_info['temp_file_count'] > self._config.max_temp_files:
            results['temp_files_cleared'] = self.clear_temp_directory()
        
        if cache_info['query_file_count'] > self._config.max_query_images:
            results['query_images_cleared'] = self.clear_old_query_images()
        
        if cache_info['needs_cleanup']:
            results['old_files_cleared'] = self.clear_old_temp_files()
            
            cache_info = self.get_cache_info()
            if cache_info['needs_cleanup']:
                results['temp_files_cleared'] += self.clear_temp_directory()
        
        self._logger.info(f"Cache limit enforcement complete: {results}")
        
        return results
    
    def auto_cleanup(self) -> bool:
        cache_info = self.get_cache_info()
        
        if not cache_info['needs_cleanup']:
            return False
        
        self._logger.info("Auto cleanup triggered")
        self.enforce_cache_limit()
        
        return True
    
    def get_cache_status_text(self) -> str:
        cache_info = self.get_cache_info()
        
        status_lines = [
            "# Cache Status",
            "",
            "## Storage Usage",
            f"- **Total Size**: {cache_info['total_size_mb']:.2f} MB / {cache_info['max_cache_size_mb']:.2f} MB",
            f"- **Usage**: {cache_info['usage_percent']:.1f}%",
            f"- **Temp Files**: {cache_info['temp_size_mb']:.2f} MB ({cache_info['temp_file_count']} files)",
            f"- **Query Images**: {cache_info['queries_size_mb']:.2f} MB ({cache_info['query_file_count']} files)",
            "",
            "## Directories",
            f"- **Temp Dir**: `{cache_info['temp_dir']}`",
            f"- **Queries Dir**: `{cache_info['queries_dir']}`",
            "",
            "## Status",
            f"- **Needs Cleanup**: {'Yes' if cache_info['needs_cleanup'] else 'No'}",
        ]
        
        return "\n".join(status_lines)


def get_cache_manager() -> CacheManager:
    return CacheManager()


class Query(Base):
    __tablename__ = 'queries'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    query_image_path = Column(String(512), nullable=False)
    query_image_hash = Column(String(64), nullable=True)
    search_mode = Column(String(20), nullable=False)
    keyword = Column(String(256), nullable=True)
    local_folder = Column(String(512), nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    
    candidates = relationship("Candidate", back_populates="query", cascade="all, delete-orphan")
    results = relationship("Result", back_populates="query", cascade="all, delete-orphan")


class Candidate(Base):
    __tablename__ = 'candidates'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    query_id = Column(Integer, ForeignKey('queries.id'), nullable=False)
    candidate_url = Column(String(1024), nullable=True)
    image_path = Column(String(512), nullable=False)
    thumbnail_url = Column(String(1024), nullable=True)
    image_hash = Column(String(64), nullable=True)
    feature_vector = Column(LargeBinary, nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    
    query = relationship("Query", back_populates="candidates")
    results = relationship("Result", back_populates="candidate", cascade="all, delete-orphan")


class Result(Base):
    __tablename__ = 'results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    query_id = Column(Integer, ForeignKey('queries.id'), nullable=False)
    candidate_id = Column(Integer, ForeignKey('candidates.id'), nullable=False)
    rank = Column(Integer, nullable=False)
    similarity = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    
    query = relationship("Query", back_populates="results")
    candidate = relationship("Candidate", back_populates="results")


class FeatureExtractor:
    _instance = None
    _initialized = False
    
    MODEL_NAME = "google/siglip-base-patch16-384"
    EMBEDDING_DIM = 768
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FeatureExtractor, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not FeatureExtractor._initialized:
            self._logger = get_logger()
            self._logger.info("Loading SigLIP model...")
            
            self.device = device
            self.model_name = self.MODEL_NAME
            self.processor = None
            self.model = None
            
            self._load_model()
            
            self._logger.info(f"Model loaded on {self.device}")
            FeatureExtractor._initialized = True
    
    def _load_model(self) -> None:
        try:
            from transformers import AutoProcessor, AutoModel
            
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            assert self.processor is not None, "Failed to load processor"
            
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            assert self.model is not None, "Failed to load model"
            
            self.model.eval()
        except Exception as e:
            raise ModelLoadError(f"Failed to load SigLIP model: {e}")
    
    def _validate_model_loaded(self) -> None:
        assert self.processor is not None, "Processor not initialized"
        assert self.model is not None, "Model not initialized"
    
    def _preprocess_image(self, image: Union[Image.Image, str]) -> Image.Image:
        if isinstance(image, str):
            validate_image_path(image)
            image = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            image = image.convert('RGB')
        else:
            raise ValidationError(f"Image must be PIL Image or file path, got {type(image)}")
        
        assert image.mode == 'RGB', f"Image mode must be RGB, got {image.mode}"
        return image
    
    def encode_image(self, image: Union[Image.Image, str]) -> Optional[np.ndarray]:
        self._validate_model_loaded()
        logger = get_logger()
        
        try:
            image = self._preprocess_image(image)
            validate_pil_image(image)
            
            inputs = self.processor(images=image, return_tensors="pt")
            assert inputs is not None, "Processor returned None"
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                assert image_features is not None, "Model returned None features"
                
                norm = image_features.norm(dim=-1, keepdim=True)
                assert (norm > 0).all(), "Feature norm is zero"
                
                image_features = image_features / norm
            
            embedding = image_features.cpu().numpy().flatten()
            
            assert embedding.shape[0] == self.EMBEDDING_DIM, \
                f"Unexpected embedding dimension: {embedding.shape[0]}"
            
            return embedding
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            return None
    
    def encode_images_batch(self, images: list, batch_size: int = 8) -> list:
        self._validate_model_loaded()
        logger = get_logger()
        
        assert isinstance(images, list), f"Images must be a list, got {type(images)}"
        assert len(images) > 0, "Images list cannot be empty"
        assert batch_size > 0, f"Batch size must be positive, got {batch_size}"
        
        all_embeddings = []
        total_batches = (len(images) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(images))
            batch = images[start_idx:end_idx]
            
            assert len(batch) > 0, "Batch cannot be empty"
            
            try:
                processed_batch = []
                for img in batch:
                    if isinstance(img, Image.Image):
                        processed_batch.append(img.convert('RGB'))
                    else:
                        logger.warning(f"Skipping non-PIL image in batch")
                        processed_batch.append(None)
                
                valid_batch = [img for img in processed_batch if img is not None]
                
                if len(valid_batch) == 0:
                    all_embeddings.extend([None] * len(batch))
                    continue
                
                inputs = self.processor(images=valid_batch, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)
                    norm = image_features.norm(dim=-1, keepdim=True)
                    image_features = image_features / norm
                
                batch_embeddings = image_features.cpu().numpy()
                
                embed_idx = 0
                for img in processed_batch:
                    if img is not None:
                        all_embeddings.append(batch_embeddings[embed_idx])
                        embed_idx += 1
                    else:
                        all_embeddings.append(None)
                        
            except Exception as e:
                logger.error(f"Error encoding batch {batch_idx + 1}/{total_batches}: {e}")
                all_embeddings.extend([None] * len(batch))
        
        assert len(all_embeddings) == len(images), \
            f"Output length mismatch: {len(all_embeddings)} vs {len(images)}"
        
        return all_embeddings
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        validate_embedding(embedding1)
        validate_embedding(embedding2)
        
        assert embedding1.shape == embedding2.shape, \
            f"Embedding shapes must match: {embedding1.shape} vs {embedding2.shape}"
        
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        assert norm1 > 0, "First embedding has zero norm"
        assert norm2 > 0, "Second embedding has zero norm"
        
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        similarity = float(np.clip(similarity, 0.0, 1.0))
        
        assert 0.0 <= similarity <= 1.0, f"Similarity out of range: {similarity}"
        
        return similarity


class ImageCrawler:
    SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    DEFAULT_MIN_SIZE = (200, 200)
    DEFAULT_MAX_THREADS = 4
    DEFAULT_LOG_LEVEL = 30
    
    def __init__(self, storage_dir: str = None, config: SearchConfig = None):
        self._logger = get_logger()
        self._config = config or DEFAULT_CONFIG
        
        if storage_dir is not None:
            assert isinstance(storage_dir, str), f"Storage dir must be a string, got {type(storage_dir)}"
        
        self.storage_dir = Path(storage_dir) if storage_dir else TEMP_DIR
        self._ensure_storage_exists()
    
    def _ensure_storage_exists(self) -> None:
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            assert self.storage_dir.exists(), f"Failed to create storage directory: {self.storage_dir}"
            assert self.storage_dir.is_dir(), f"Storage path is not a directory: {self.storage_dir}"
        except Exception as e:
            raise CrawlerError(f"Failed to initialize storage directory: {e}")
    
    def clear_storage(self) -> None:
        self._logger.info(f"Clearing storage directory: {self.storage_dir}")
        
        try:
            if self.storage_dir.exists():
                file_count = len(list(self.storage_dir.iterdir()))
                shutil.rmtree(self.storage_dir)
                self._logger.info(f"Removed {file_count} files from storage")
            
            self._ensure_storage_exists()
        except Exception as e:
            raise CrawlerError(f"Failed to clear storage: {e}")
    
    def _validate_crawl_params(self, keyword: str, max_num: int) -> Tuple[str, int]:
        validate_keyword(keyword)
        validate_max_num(max_num)
        
        keyword = sanitize_keyword(keyword)
        max_num = int(max_num)
        
        return keyword, max_num
    
    def _collect_downloaded_images(self) -> List[str]:
        image_files = []
        
        assert self.storage_dir.exists(), f"Storage directory does not exist: {self.storage_dir}"
        
        for file in self.storage_dir.iterdir():
            if not file.is_file():
                continue
            
            suffix = file.suffix.lower()
            if suffix not in self.SUPPORTED_EXTENSIONS:
                self._logger.debug(f"Skipping unsupported file: {file}")
                continue
            
            file_path = str(file)
            
            try:
                file_size = file.stat().st_size
                if file_size == 0:
                    self._logger.warning(f"Skipping empty file: {file}")
                    continue
                
                image_files.append(file_path)
            except Exception as e:
                self._logger.warning(f"Error checking file {file}: {e}")
                continue
        
        return image_files
    
    def crawl_images(self, keyword: str, max_num: int = 20) -> List[str]:
        keyword, max_num = self._validate_crawl_params(keyword, max_num)
        
        self._logger.info(f"Starting image crawl for keyword: '{keyword}' (max: {max_num})")
        
        try:
            from icrawler.builtin import BingImageCrawler
        except ImportError as e:
            raise CrawlerError(f"icrawler package not installed: {e}")
        
        self.clear_storage()
        
        start_time = time.time()
        
        try:
            crawler = BingImageCrawler(
                storage={'root_dir': str(self.storage_dir)},
                downloader_threads=self._config.max_crawl_threads,
                log_level=self.DEFAULT_LOG_LEVEL
            )
            
            crawler.crawl(
                keyword=keyword,
                max_num=max_num,
                min_size=self._config.min_image_size,
                max_size=None
            )
        except Exception as e:
            raise CrawlerError(f"Crawling failed: {e}")
        
        elapsed_time = time.time() - start_time
        image_files = self._collect_downloaded_images()
        
        self._logger.info(
            f"Crawl completed: {len(image_files)} images in {elapsed_time:.2f}s"
        )
        
        return image_files
    
    def get_storage_info(self) -> Dict[str, Any]:
        if not self.storage_dir.exists():
            return {'exists': False, 'file_count': 0, 'total_size': 0}
        
        files = list(self.storage_dir.iterdir())
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        
        return {
            'exists': True,
            'path': str(self.storage_dir),
            'file_count': len(files),
            'total_size': total_size,
            'total_size_mb': total_size / (1024 * 1024)
        }


class DatabaseManager:
    HASH_CHUNK_SIZE = 4096
    MAX_QUERY_HISTORY = 1000
    
    def __init__(self, db_path: str = None):
        self._logger = get_logger()
        
        if db_path is not None:
            assert isinstance(db_path, str), f"DB path must be a string, got {type(db_path)}"
        
        db_path = Path(db_path) if db_path else DB_PATH
        
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            assert db_path.parent.exists(), f"Failed to create DB directory: {db_path.parent}"
        except Exception as e:
            raise DatabaseError(f"Failed to initialize database directory: {e}")
        
        self.db_path = str(db_path)
        self._initialize_database()
        
        self._logger.info(f"Database initialized: {self.db_path}")
    
    def _initialize_database(self) -> None:
        try:
            self.engine = create_engine(f'sqlite:///{self.db_path}', echo=False)
            assert self.engine is not None, "Failed to create database engine"
            
            Base.metadata.create_all(self.engine)
            self.SessionLocal = sessionmaker(bind=self.engine)
            assert self.SessionLocal is not None, "Failed to create session factory"
        except Exception as e:
            raise DatabaseError(f"Failed to initialize database: {e}")
    
    @contextmanager
    def get_session(self):
        assert self.SessionLocal is not None, "Session factory not initialized"
        
        session = self.SessionLocal()
        assert session is not None, "Failed to create session"
        
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self._logger.error(f"Database session error: {e}")
            raise DatabaseError(f"Database operation failed: {e}")
        finally:
            session.close()
    
    @staticmethod
    def compute_image_hash(image_path: str) -> str:
        assert isinstance(image_path, str), f"Image path must be a string, got {type(image_path)}"
        assert len(image_path) > 0, "Image path cannot be empty"
        assert os.path.exists(image_path), f"Image path does not exist: {image_path}"
        assert os.path.isfile(image_path), f"Image path is not a file: {image_path}"
        
        md5 = hashlib.md5()
        
        with open(image_path, 'rb') as f:
            for chunk in iter(lambda: f.read(DatabaseManager.HASH_CHUNK_SIZE), b""):
                md5.update(chunk)
        
        hash_result = md5.hexdigest()
        assert len(hash_result) == 32, f"Unexpected hash length: {len(hash_result)}"
        
        return hash_result
    
    @staticmethod
    def serialize_feature_vector(vector: np.ndarray) -> bytes:
        assert isinstance(vector, np.ndarray), f"Vector must be numpy array, got {type(vector)}"
        assert vector.size > 0, "Vector cannot be empty"
        
        serialized = vector.astype(np.float32).tobytes()
        assert len(serialized) > 0, "Serialization produced empty result"
        
        return serialized
    
    @staticmethod
    def deserialize_feature_vector(data: bytes) -> np.ndarray:
        assert isinstance(data, bytes), f"Data must be bytes, got {type(data)}"
        assert len(data) > 0, "Data cannot be empty"
        
        vector = np.frombuffer(data, dtype=np.float32)
        assert vector.size > 0, "Deserialization produced empty vector"
        
        return vector
    
    def _validate_search_mode(self, search_mode: str) -> str:
        assert isinstance(search_mode, str), f"Search mode must be a string, got {type(search_mode)}"
        
        search_mode = search_mode.lower().strip()
        valid_modes = [mode.value for mode in SearchMode]
        
        assert search_mode in valid_modes, \
            f"Invalid search mode: {search_mode}. Valid modes: {valid_modes}"
        
        return search_mode
    
    def create_query(self, query_image_path: str, search_mode: str, 
                     keyword: Optional[str] = None, local_folder: Optional[str] = None) -> int:
        assert isinstance(query_image_path, str), \
            f"Query image path must be a string, got {type(query_image_path)}"
        assert len(query_image_path) > 0, "Query image path cannot be empty"
        
        search_mode = self._validate_search_mode(search_mode)
        
        if search_mode == SearchMode.ONLINE.value:
            assert keyword is not None, "Keyword is required for online search"
        elif search_mode == SearchMode.LOCAL.value:
            assert local_folder is not None, "Local folder is required for local search"
        
        with self.get_session() as session:
            image_hash = self.compute_image_hash(query_image_path)
            
            query = Query(
                query_image_path=query_image_path,
                query_image_hash=image_hash,
                search_mode=search_mode,
                keyword=keyword,
                local_folder=local_folder
            )
            
            session.add(query)
            session.flush()
            
            query_id = query.id
            assert query_id is not None, "Failed to get query ID"
            assert query_id > 0, f"Invalid query ID: {query_id}"
            
            self._logger.info(f"Created query record: ID={query_id}")
            
            return query_id
    
    def save_query_results(self, query_id: int, results: List[Tuple[str, float]], 
                          feature_vectors: Optional[List[np.ndarray]] = None) -> int:
        assert isinstance(query_id, int), f"Query ID must be an integer, got {type(query_id)}"
        assert query_id > 0, f"Query ID must be positive, got {query_id}"
        assert isinstance(results, list), f"Results must be a list, got {type(results)}"
        
        if feature_vectors is not None:
            assert isinstance(feature_vectors, list), \
                f"Feature vectors must be a list, got {type(feature_vectors)}"
            assert len(feature_vectors) >= len(results), \
                f"Feature vectors length mismatch: {len(feature_vectors)} < {len(results)}"
        
        saved_count = 0
        
        with self.get_session() as session:
            for rank, (image_path, similarity) in enumerate(results, start=1):
                assert isinstance(image_path, str), \
                    f"Image path must be a string, got {type(image_path)}"
                assert isinstance(similarity, (int, float)), \
                    f"Similarity must be a number, got {type(similarity)}"
                
                feature_vec = feature_vectors[rank-1] if feature_vectors else None
                
                image_hash = None
                if os.path.exists(image_path):
                    try:
                        image_hash = self.compute_image_hash(image_path)
                    except Exception as e:
                        self._logger.warning(f"Failed to compute hash for {image_path}: {e}")
                
                serialized_vector = None
                if feature_vec is not None:
                    try:
                        serialized_vector = self.serialize_feature_vector(feature_vec)
                    except Exception as e:
                        self._logger.warning(f"Failed to serialize vector: {e}")
                
                candidate = Candidate(
                    query_id=query_id,
                    image_path=image_path,
                    image_hash=image_hash,
                    feature_vector=serialized_vector
                )
                session.add(candidate)
                session.flush()
                
                assert candidate.id is not None, "Failed to get candidate ID"
                
                result = Result(
                    query_id=query_id,
                    candidate_id=candidate.id,
                    rank=rank,
                    similarity=float(similarity)
                )
                session.add(result)
                saved_count += 1
        
        self._logger.info(f"Saved {saved_count} results for query {query_id}")
        
        return saved_count
    
    def get_query_history(self, limit: int = 50) -> List[Dict]:
        assert isinstance(limit, int), f"Limit must be an integer, got {type(limit)}"
        assert limit > 0, f"Limit must be positive, got {limit}"
        assert limit <= self.MAX_QUERY_HISTORY, \
            f"Limit exceeds maximum: {limit} > {self.MAX_QUERY_HISTORY}"
        
        with self.get_session() as session:
            queries = session.query(Query).order_by(desc(Query.created_at)).limit(limit).all()
            
            history = []
            for q in queries:
                assert q.id is not None, "Query ID is None"
                assert q.created_at is not None, "Query created_at is None"
                
                history_entry = {
                    'id': q.id,
                    'image_path': q.query_image_path,
                    'mode': q.search_mode,
                    'keyword': q.keyword,
                    'local_folder': q.local_folder,
                    'created_at': q.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                    'num_results': len(q.results) if q.results else 0
                }
                history.append(history_entry)
            
            assert len(history) <= limit, \
                f"History length exceeds limit: {len(history)} > {limit}"
            
            return history
    
    def get_query_results(self, query_id: int) -> List[Tuple[str, float]]:
        query_id = validate_query_id(query_id)
        
        with self.get_session() as session:
            query_exists = session.query(Query).filter(Query.id == query_id).first()
            if query_exists is None:
                self._logger.warning(f"Query ID {query_id} not found")
                return []
            
            results = session.query(Result).filter(
                Result.query_id == query_id
            ).order_by(Result.rank).all()
            
            output = []
            for r in results:
                assert r.candidate_id is not None, "Result candidate_id is None"
                assert r.similarity is not None, "Result similarity is None"
                
                candidate = session.query(Candidate).filter(
                    Candidate.id == r.candidate_id
                ).first()
                
                if candidate is not None:
                    assert candidate.image_path is not None, "Candidate image_path is None"
                    output.append((candidate.image_path, float(r.similarity)))
            
            self._logger.debug(f"Retrieved {len(output)} results for query {query_id}")
            
            return output
    
    def get_statistics(self) -> Dict[str, Any]:
        with self.get_session() as session:
            total_queries = session.query(Query).count()
            total_candidates = session.query(Candidate).count()
            total_results = session.query(Result).count()
            
            assert total_queries >= 0, f"Invalid query count: {total_queries}"
            assert total_candidates >= 0, f"Invalid candidate count: {total_candidates}"
            assert total_results >= 0, f"Invalid result count: {total_results}"
            
            online_queries = session.query(Query.keyword).filter(
                Query.keyword.isnot(None)
            ).all()
            
            keywords = [q.keyword for q in online_queries if q.keyword]
            keyword_counts: Dict[str, int] = {}
            
            for kw in keywords:
                assert isinstance(kw, str), f"Keyword must be string, got {type(kw)}"
                keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
            
            top_keywords = sorted(
                keyword_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            statistics = {
                'total_queries': total_queries,
                'total_candidates': total_candidates,
                'total_results': total_results,
                'top_keywords': top_keywords,
                'unique_keywords': len(keyword_counts),
                'avg_results_per_query': total_results / max(total_queries, 1)
            }
            
            return statistics
    
    def delete_query(self, query_id: int) -> bool:
        query_id = validate_query_id(query_id)
        
        with self.get_session() as session:
            query = session.query(Query).filter(Query.id == query_id).first()
            
            if query is None:
                self._logger.warning(f"Query {query_id} not found for deletion")
                return False
            
            session.delete(query)
            self._logger.info(f"Deleted query {query_id}")
            
            return True
    
    def clear_old_queries(self, keep_recent: int = 100) -> int:
        assert isinstance(keep_recent, int), f"keep_recent must be an integer, got {type(keep_recent)}"
        assert keep_recent >= 0, f"keep_recent must be non-negative, got {keep_recent}"
        
        with self.get_session() as session:
            recent_ids = session.query(Query.id).order_by(
                desc(Query.created_at)
            ).limit(keep_recent).all()
            
            recent_ids = [q.id for q in recent_ids]
            
            if recent_ids:
                deleted = session.query(Query).filter(
                    ~Query.id.in_(recent_ids)
                ).delete(synchronize_session=False)
            else:
                deleted = session.query(Query).delete(synchronize_session=False)
            
            self._logger.info(f"Cleared {deleted} old queries, kept {keep_recent} recent")
            
            return deleted


class SimilarityEngine:
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    DEFAULT_BATCH_SIZE = 8
    MAX_RESULTS = 1000
    
    def __init__(self, use_database: bool = True, config: SearchConfig = None):
        assert isinstance(use_database, bool), \
            f"use_database must be a boolean, got {type(use_database)}"
        
        self._logger = get_logger()
        self._config = config or DEFAULT_CONFIG
        
        self.extractor = get_feature_extractor()
        assert self.extractor is not None, "Failed to initialize feature extractor"
        
        self.use_database = use_database
        self.db_manager = get_db_manager() if use_database else None
        
        if use_database:
            assert self.db_manager is not None, "Failed to initialize database manager"
        
        self._logger.info(f"SimilarityEngine initialized (database: {use_database})")
    
    def _validate_query_image(self, query_image: Image.Image) -> None:
        validate_pil_image(query_image)
    
    def _load_single_image(self, img_path: str) -> Optional[Image.Image]:
        assert isinstance(img_path, str), f"Image path must be string, got {type(img_path)}"
        
        try:
            if not os.path.exists(img_path):
                self._logger.warning(f"Image file not found: {img_path}")
                return None
            
            img = Image.open(img_path)
            assert img is not None, f"Failed to open image: {img_path}"
            
            img = img.convert('RGB')
            assert img.mode == 'RGB', f"Failed to convert image to RGB: {img_path}"
            
            return img
        except Exception as e:
            self._logger.warning(f"Error loading image {img_path}: {e}")
            return None
    
    def _load_and_encode_images(self, image_paths: List[str]) -> Tuple[List[str], List[np.ndarray]]:
        assert isinstance(image_paths, list), \
            f"Image paths must be a list, got {type(image_paths)}"
        
        if len(image_paths) == 0:
            self._logger.warning("Empty image paths list")
            return [], []
        
        self._logger.info(f"Loading {len(image_paths)} images...")
        
        loaded_images = []
        valid_paths = []
        failed_count = 0
        
        for img_path in image_paths:
            img = self._load_single_image(img_path)
            if img is not None:
                loaded_images.append(img)
                valid_paths.append(img_path)
            else:
                failed_count += 1
        
        if failed_count > 0:
            self._logger.warning(f"Failed to load {failed_count} images")
        
        if len(loaded_images) == 0:
            self._logger.error("No images could be loaded")
            return [], []
        
        self._logger.info(f"Encoding {len(loaded_images)} images...")
        
        batch_embeddings = self.extractor.encode_images_batch(
            loaded_images, 
            batch_size=self._config.batch_size
        )
        
        assert len(batch_embeddings) == len(loaded_images), \
            f"Embedding count mismatch: {len(batch_embeddings)} vs {len(loaded_images)}"
        
        return valid_paths, batch_embeddings
    
    def _compute_and_rank_similarities(
        self, 
        query_embedding: np.ndarray, 
        valid_paths: List[str], 
        batch_embeddings: List[np.ndarray],
        min_threshold: float = 0.0
    ) -> List[Tuple[str, float]]:
        validate_embedding(query_embedding)
        
        assert isinstance(valid_paths, list), \
            f"Valid paths must be a list, got {type(valid_paths)}"
        assert isinstance(batch_embeddings, list), \
            f"Batch embeddings must be a list, got {type(batch_embeddings)}"
        assert len(valid_paths) == len(batch_embeddings), \
            f"Length mismatch: {len(valid_paths)} vs {len(batch_embeddings)}"
        assert 0.0 <= min_threshold <= 1.0, \
            f"Threshold must be between 0 and 1, got {min_threshold}"
        
        results = []
        skipped_count = 0
        
        for img_path, img_embedding in zip(valid_paths, batch_embeddings):
            if img_embedding is None:
                skipped_count += 1
                continue
            
            try:
                similarity = self.extractor.compute_similarity(query_embedding, img_embedding)
                
                if similarity >= min_threshold:
                    results.append((img_path, similarity))
            except Exception as e:
                self._logger.warning(f"Error computing similarity for {img_path}: {e}")
                skipped_count += 1
        
        if skipped_count > 0:
            self._logger.debug(f"Skipped {skipped_count} images in similarity computation")
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        if len(results) > self.MAX_RESULTS:
            results = results[:self.MAX_RESULTS]
            self._logger.info(f"Truncated results to {self.MAX_RESULTS}")
        
        return results
    
    def _save_results_to_database(
        self, 
        query_id: Optional[int], 
        results: List[Tuple[str, float]], 
        batch_embeddings: List[np.ndarray]
    ) -> None:
        if not self.use_database:
            return
        
        if query_id is None:
            self._logger.warning("Cannot save results: query_id is None")
            return
        
        assert self.db_manager is not None, "Database manager not initialized"
        
        try:
            self.db_manager.save_query_results(
                query_id=query_id,
                results=results,
                feature_vectors=batch_embeddings
            )
            self._logger.info(f"Saved {len(results)} results to database")
        except Exception as e:
            self._logger.error(f"Failed to save results to database: {e}")
    
    def _scan_directory_for_images(self, directory: str) -> List[str]:
        validate_directory_path(directory)
        
        dir_path = Path(directory)
        assert dir_path.is_dir(), f"Not a directory: {directory}"
        
        self._logger.info(f"Scanning directory: {directory}")
        
        image_paths = []
        
        for file_path in dir_path.rglob('*'):
            if not file_path.is_file():
                continue
            
            suffix = file_path.suffix.lower()
            if suffix not in self.SUPPORTED_EXTENSIONS:
                continue
            
            image_paths.append(str(file_path))
        
        self._logger.info(f"Found {len(image_paths)} images in directory")
        
        return image_paths
    
    def search_online(
        self, 
        query_image: Image.Image, 
        keyword: str, 
        max_crawl: int = 20, 
        query_image_path: str = None
    ) -> List[Tuple[str, float]]:
        self._validate_query_image(query_image)
        validate_keyword(keyword)
        validate_max_num(max_crawl)
        
        keyword = sanitize_keyword(keyword)
        max_crawl = int(max_crawl)
        
        self._logger.info(f"Starting online search: keyword='{keyword}', max={max_crawl}")
        
        query_id = None
        if self.use_database and query_image_path:
            assert isinstance(query_image_path, str), \
                f"Query image path must be string, got {type(query_image_path)}"
            
            query_id = self.db_manager.create_query(
                query_image_path=query_image_path,
                search_mode=SearchMode.ONLINE.value,
                keyword=keyword
            )
            self._logger.info(f"Created query record: ID={query_id}")
        
        query_embedding = self.extractor.encode_image(query_image)
        if query_embedding is None:
            self._logger.error("Failed to encode query image")
            return []
        
        crawler = ImageCrawler(config=self._config)
        image_paths = crawler.crawl_images(keyword, max_num=max_crawl)
        
        if not image_paths:
            self._logger.warning("No images crawled")
            return []
        
        valid_paths, batch_embeddings = self._load_and_encode_images(image_paths)
        
        if not valid_paths:
            self._logger.warning("No valid images after loading")
            return []
        
        results = self._compute_and_rank_similarities(
            query_embedding, 
            valid_paths, 
            batch_embeddings,
            min_threshold=self._config.min_similarity_threshold
        )
        
        self._save_results_to_database(query_id, results, batch_embeddings)
        
        self._logger.info(f"Online search completed: {len(results)} results")
        
        return results
    
    def search_local(
        self, 
        query_image: Image.Image, 
        directory: str, 
        query_image_path: str = None
    ) -> List[Tuple[str, float]]:
        self._validate_query_image(query_image)
        validate_directory_path(directory)
        
        directory = normalize_path(directory)
        
        self._logger.info(f"Starting local search: directory='{directory}'")
        
        query_id = None
        if self.use_database and query_image_path:
            assert isinstance(query_image_path, str), \
                f"Query image path must be string, got {type(query_image_path)}"
            
            query_id = self.db_manager.create_query(
                query_image_path=query_image_path,
                search_mode=SearchMode.LOCAL.value,
                local_folder=directory
            )
            self._logger.info(f"Created query record: ID={query_id}")
        
        query_embedding = self.extractor.encode_image(query_image)
        if query_embedding is None:
            self._logger.error("Failed to encode query image")
            return []
        
        image_paths = self._scan_directory_for_images(directory)
        
        if not image_paths:
            self._logger.warning("No images found in directory")
            return []
        
        valid_paths, batch_embeddings = self._load_and_encode_images(image_paths)
        
        if not valid_paths:
            self._logger.warning("No valid images after loading")
            return []
        
        results = self._compute_and_rank_similarities(
            query_embedding, 
            valid_paths, 
            batch_embeddings,
            min_threshold=self._config.min_similarity_threshold
        )
        
        self._save_results_to_database(query_id, results, batch_embeddings)
        
        self._logger.info(f"Local search completed: {len(results)} results")
        
        return results


_extractor = None
_db_manager = None
_engine = None


def get_feature_extractor() -> FeatureExtractor:
    global _extractor
    if _extractor is None:
        _extractor = FeatureExtractor()
    return _extractor


def get_db_manager() -> DatabaseManager:
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def get_engine() -> SimilarityEngine:
    global _engine
    if _engine is None:
        _engine = SimilarityEngine()
    return _engine


def save_query_image(query_image: Image.Image) -> str:
    QUERIES_DIR.mkdir(parents=True, exist_ok=True)
    query_filename = f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
    query_image_path = str(QUERIES_DIR / query_filename)
    query_image.save(query_image_path)
    return query_image_path


def search_similar_images(query_image: Image.Image, search_mode: str, keyword: str, 
                          local_folder: str, max_num: int) -> Tuple[List[str], str]:
    if query_image is None:
        return [], "Please upload a query image first."
    
    query_image_path = save_query_image(query_image)
    engine = get_engine()
    
    start_time = time.time()
    
    try:
        if search_mode == "local":
            if not local_folder or local_folder.strip() == "":
                return [], "Please enter a local folder path."
            if not os.path.exists(local_folder):
                return [], f"Folder does not exist: {local_folder}"
            
            results = engine.search_local(
                query_image=query_image,
                directory=local_folder.strip(),
                query_image_path=query_image_path
            )
            search_info = f"Local folder: {local_folder.strip()}"
        else:
            if not keyword or keyword.strip() == "":
                return [], "Please enter a search keyword for online search."
            
            results = engine.search_online(
                query_image=query_image,
                keyword=keyword.strip(),
                max_crawl=int(max_num),
                query_image_path=query_image_path
            )
            search_info = f"Online search - Keyword: {keyword.strip()}"
        
        if not results:
            elapsed_time = time.time() - start_time
            return [], f"No matching images found. Search scope: {search_info}. Time: {elapsed_time:.2f}s"
        
        elapsed_time = time.time() - start_time
        
        gallery_output = [img_path for img_path, _ in results]
        status_details = [f"{idx}. Similarity: {similarity:.4f}" for idx, (_, similarity) in enumerate(results, 1)]
        
        status_msg = (
            f"Found {len(results)} similar images!\n"
            f"Search scope: {search_info}\n"
            f"Time: {elapsed_time:.2f}s ({len(results)/elapsed_time:.2f} images/sec)\n\n"
            f"Similarity ranking (Top 10):\n" + 
            "\n".join(status_details[:10])
        )
        return gallery_output, status_msg
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        return [], f"Search failed: {str(e)}. Time: {elapsed_time:.2f}s"


def load_query_history() -> str:
    db_manager = get_db_manager()
    history = db_manager.get_query_history(limit=50)
    
    if not history:
        return "No query history yet."
    
    formatted = "# Query History\n\n"
    for h in history:
        formatted += f"### Query ID: {h['id']}\n"
        formatted += f"- **Time**: {h['created_at']}\n"
        formatted += f"- **Mode**: {h['mode']}\n"
        if h['keyword']:
            formatted += f"- **Keyword**: {h['keyword']}\n"
        if h['local_folder']:
            formatted += f"- **Folder**: {h['local_folder']}\n"
        formatted += f"- **Results**: {h['num_results']}\n"
        formatted += f"- **Query Image**: {h['image_path']}\n\n"
        formatted += "---\n\n"
    
    return formatted


def load_statistics() -> str:
    db_manager = get_db_manager()
    stats = db_manager.get_statistics()
    
    formatted = "# System Statistics\n\n"
    formatted += f"## Overview\n"
    formatted += f"- **Total Queries**: {stats['total_queries']}\n"
    formatted += f"- **Total Candidates**: {stats['total_candidates']}\n"
    formatted += f"- **Total Results**: {stats['total_results']}\n\n"
    
    if stats['top_keywords']:
        formatted += f"## Top Keywords (Top 10)\n"
        for idx, (keyword, count) in enumerate(stats['top_keywords'], 1):
            formatted += f"{idx}. **{keyword}** - {count} times\n"
    else:
        formatted += "No keyword data yet.\n"
    
    return formatted


def view_query_results(query_id_str: str) -> Tuple[List[str], str]:
    try:
        query_id = int(query_id_str)
        db_manager = get_db_manager()
        results = db_manager.get_query_results(query_id)
        
        if not results:
            return [], f"Query ID {query_id} has no results."
        
        gallery = [img_path for img_path, _ in results]
        info = f"Query ID {query_id} has {len(results)} results\n\n"
        info += "**Similarity Ranking**:\n"
        for idx, (img_path, sim) in enumerate(results[:10], 1):
            info += f"{idx}. Similarity: {sim:.4f}\n"
        
        return gallery, info
    except ValueError:
        return [], "Please enter a valid Query ID (integer)."
    except Exception as e:
        return [], f"Error: {str(e)}"


def get_cache_status() -> str:
    try:
        cache_manager = get_cache_manager()
        return cache_manager.get_cache_status_text()
    except Exception as e:
        return f"Error getting cache status: {str(e)}"


def clear_temp_cache() -> str:
    try:
        cache_manager = get_cache_manager()
        cleared_count = cache_manager.clear_temp_directory()
        return f"Successfully cleared {cleared_count} items from temp directory."
    except Exception as e:
        return f"Error clearing temp cache: {str(e)}"


def clear_old_temp_files() -> str:
    try:
        cache_manager = get_cache_manager()
        cleared_count = cache_manager.clear_old_temp_files(max_age_hours=24)
        return f"Successfully cleared {cleared_count} old temp files (older than 24 hours)."
    except Exception as e:
        return f"Error clearing old temp files: {str(e)}"


def clear_old_query_images(keep_recent: int) -> str:
    try:
        keep_recent = int(keep_recent)
        cache_manager = get_cache_manager()
        cleared_count = cache_manager.clear_old_query_images(keep_recent=keep_recent)
        return f"Successfully cleared {cleared_count} old query images (kept {keep_recent} recent)."
    except Exception as e:
        return f"Error clearing old query images: {str(e)}"


def enforce_cache_limits() -> str:
    try:
        cache_manager = get_cache_manager()
        results = cache_manager.enforce_cache_limit()
        
        lines = [
            "Cache limit enforcement completed:",
            f"- Temp files cleared: {results['temp_files_cleared']}",
            f"- Query images cleared: {results['query_images_cleared']}",
            f"- Old files cleared: {results['old_files_cleared']}"
        ]
        
        return "\n".join(lines)
    except Exception as e:
        return f"Error enforcing cache limits: {str(e)}"
