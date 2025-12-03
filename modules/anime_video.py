import os
import ssl
import cv2
import numpy as np
from PIL import Image
import torch
import tempfile
import uuid
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Set, Tuple, Optional

from modules.shared import state

ssl._create_default_https_context = ssl._create_unverified_context

device: str = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Video2Anime] Using device: {device}")

STYLE_VARIANTS: Dict[str, str] = {
    "face_paint_512_v2": "face_paint_512_v2 (Portrait, smooth skin)",
    "face_paint_512_v1": "face_paint_512_v1 (Classic portrait style)",
    "paprika": "paprika (Anime show style)",
}

VALID_VARIANTS: Tuple[str, ...] = tuple(STYLE_VARIANTS.keys())
DEFAULT_VARIANT: str = "face_paint_512_v2"

_downloaded_variants: Set[str] = set()
_anime_models: Dict[str, torch.nn.Module] = {}


def _clamp(value: float, minimum: float, maximum: float) -> float:
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def _normalize_variant(variant: Optional[str]) -> str:
    if not variant:
        result = DEFAULT_VARIANT
    elif variant not in VALID_VARIANTS:
        result = DEFAULT_VARIANT
    else:
        result = variant

    assert result in VALID_VARIANTS
    return result


def is_variant_downloaded(variant: str) -> bool:
    normalized = _normalize_variant(variant)
    if normalized == DEFAULT_VARIANT:
        return True
    return normalized in _downloaded_variants


def get_download_status() -> Dict[str, bool]:
    return {variant: is_variant_downloaded(variant) for variant in VALID_VARIANTS}


def download_variant(variant: str) -> str:
    normalized = _normalize_variant(variant)
    try:
        _load_animegan2_internal(normalized, force_reload=True)
        _downloaded_variants.add(normalized)
        return f"{normalized} downloaded."
    except Exception as exc:
        return f"Failed to download {normalized}: {exc}"


def _load_animegan2_internal(
    variant: str,
    force_reload: bool = False,
) -> torch.nn.Module:
    normalized = _normalize_variant(variant)

    if not force_reload and normalized in _anime_models:
        return _anime_models[normalized]

    print(
        f"[Video2Anime] Loading AnimeGANv2 generator "
        f"'{normalized}' from bryandlee/animegan2-pytorch..."
    )

    model = (
        torch.hub.load(
            "bryandlee/animegan2-pytorch:main",
            "generator",
            pretrained=normalized,
        )
        .to(device)
        .eval()
    )

    _anime_models[normalized] = model
    print(f"[Video2Anime] AnimeGANv2 generator '{normalized}' loaded.")
    return model


def load_animegan2(variant: Optional[str] = None) -> torch.nn.Module:
    return _load_animegan2_internal(_normalize_variant(variant), force_reload=False)


def animegan_frame(
    frame_bgr: np.ndarray,
    style_variant: Optional[str] = None,
) -> np.ndarray:
    assert isinstance(frame_bgr, np.ndarray)
    assert frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3

    model = load_animegan2(style_variant)

    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    img_np = np.asarray(pil_img, dtype=np.float32) / 255.0
    img_np = img_np.transpose(2, 0, 1)
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)
    img_tensor = img_tensor * 2.0 - 1.0

    with torch.no_grad():
        out_tensor = model(img_tensor)

    out_tensor = out_tensor[0].detach().cpu().clamp(-1.0, 1.0)
    out_tensor = (out_tensor + 1.0) / 2.0
    out_np = (out_tensor * 255.0).byte().numpy()
    out_np = out_np.transpose(1, 2, 0)

    out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
    return out_bgr


def _create_video_writer(
    path: str,
    fps: float,
    size: Tuple[int, int],
) -> cv2.VideoWriter:
    assert fps > 0
    width, height = size
    assert width > 0 and height > 0

    tried: list[str] = []

    for fourcc_str in ("avc1", "H264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(path, fourcc, fps, size)
        if writer.isOpened():
            print(f"[Video2Anime] Using codec: {fourcc_str}")
            return writer
        writer.release()
        tried.append(fourcc_str)

    raise RuntimeError(f"Failed to create VideoWriter. Tried codecs: {tried}")


def _merge_audio(original_path: str, silent_video_path: str) -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        return silent_video_path

    original_path = str(original_path)
    silent_video_path = str(silent_video_path)
    temp_output = silent_video_path + ".tmp.mp4"

    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        silent_video_path,
        "-i",
        original_path,
        "-c",
        "copy",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        temp_output,
    ]

    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception:
        if os.path.exists(temp_output):
            try:
                os.remove(temp_output)
            except Exception:
                pass
        return silent_video_path

    try:
        os.replace(temp_output, silent_video_path)
    except Exception:
        if os.path.exists(temp_output):
            try:
                os.remove(temp_output)
            except Exception:
                pass

    return silent_video_path


def animegan_video(
    video_path: str,
    max_side: int = 720,
    style_variant: Optional[str] = None,
    strength: float = 1.0,
    smoothing: float = 0.0, 
) -> str:
    if not isinstance(video_path, str) or not os.path.exists(video_path):
        raise RuntimeError(f"Video file does not exist: {video_path}")

    style_variant = _normalize_variant(style_variant)

    if style_variant != DEFAULT_VARIANT and not is_variant_downloaded(style_variant):
        raise RuntimeError(
            f"Style '{style_variant}' is not downloaded yet. "
            f"Please click the Download button first."
        )

    try:
        strength_value = float(strength)
    except Exception:
        strength_value = 1.0
    strength_value = _clamp(strength_value, 0.0, 1.0)

    try:
        smoothing_value = float(smoothing)
    except Exception:
        smoothing_value = 0.0
    smoothing_value = _clamp(smoothing_value, 0.0, 1.0)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print(f"[Video2Anime] Total frames: {total_frames}, FPS: {fps}")

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError("No valid frame found in the video.")

    height, width = frame.shape[:2]
    assert height > 0 and width > 0

    scale = 1.0
    if max(height, width) > max_side:
        scale = max_side / float(max(height, width))
        new_width = int(width * scale)
        new_height = int(height * scale)
    else:
        new_width, new_height = width, height

    assert new_width > 0 and new_height > 0
    print(
        f"[Video2Anime] Processing resolution: {new_width}x{new_height} "
        f"(original {width}x{height})"
    )

    tmpdir = tempfile.mkdtemp(prefix="animegan2_video_")
    silent_out_path = os.path.join(tmpdir, f"anime_{uuid.uuid4().hex}.mp4")

    writer = _create_video_writer(silent_out_path, float(fps), (new_width, new_height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_idx = 0
    prev_output: Optional[np.ndarray] = None

    while True:
        if state.interrupted:
            print("[Video2Anime] Interrupted by user. Stopping processing.")
            break

        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if scale != 1.0:
            frame = cv2.resize(
                frame,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA,
            )

        try:
            stylized = animegan_frame(frame, style_variant=style_variant)
            if strength_value < 1.0:
                out_frame = cv2.addWeighted(
                    stylized,
                    strength_value,
                    frame,
                    1.0 - strength_value,
                    0.0,
                )
            else:
                out_frame = stylized
        except Exception as exc:
            print(
                f"[Video2Anime][Warning] Failed to process frame {frame_idx}. "
                f"Using original frame. Error: {exc}"
            )
            out_frame = frame

        if smoothing_value > 0.0 and prev_output is not None:
            out_frame = cv2.addWeighted(
                out_frame,
                1.0 - smoothing_value,
                prev_output,
                smoothing_value,
                0.0,
            )

        writer.write(out_frame)
        prev_output = out_frame

        frame_idx += 1
        if total_frames > 0 and frame_idx % 30 == 0:
            print(f"[Video2Anime] Processed {frame_idx}/{total_frames} frames")

    cap.release()
    writer.release()

    final_path = _merge_audio(video_path, silent_out_path)
    print("[Video2Anime] Video processing finished:", final_path)
    return os.path.abspath(final_path)


def gradio_animegan(
    video,
    max_side,
    style_variant: Optional[str] = None,
    strength: float = 1.0,
    smoothing: float = 0.0,
):
    if video is None:
        raise RuntimeError("Please upload a video file first.")

    if isinstance(video, dict) and "name" in video:
        video_path = str(video["name"])
    elif isinstance(video, (str, Path)):
        video_path = str(video)
    else:
        raise RuntimeError("Invalid video input.")

    try:
        max_side_int = int(max_side)
    except Exception:
        max_side_int = 720

    max_side_int = max(1, max_side_int)
    style_variant = _normalize_variant(style_variant)

    try:
        strength_value = float(strength)
    except Exception:
        strength_value = 1.0
    strength_value = _clamp(strength_value, 0.0, 1.0)

    try:
        smoothing_value = float(smoothing)
    except Exception:
        smoothing_value = 0.0
    smoothing_value = _clamp(smoothing_value, 0.0, 1.0)

    state.interrupted = False
    state.job = "AnimeGANv2 video stylization"

    return animegan_video(
        video_path=video_path,
        max_side=max_side_int,
        style_variant=style_variant,
        strength=strength_value,
        smoothing=smoothing_value,
    )
