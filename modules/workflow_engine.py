import json
from dataclasses import dataclass, field
from typing import Any, Dict, List
from contextlib import closing

from modules import processing
import modules.shared as shared
from modules.shared import opts

import base64
import io
from PIL import Image
from modules import images


try:
    from modules import prompt_optimizer_llm
except Exception:
    prompt_optimizer_llm = None


@dataclass
class WorkflowNode:
    id: str
    type: str
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowEdge:
    src: str
    src_port: str
    dst: str
    dst_port: str


@dataclass
class WorkflowGraph:
    nodes: Dict[str, WorkflowNode]
    edges: List[WorkflowEdge]


class WorkflowContext:
    def __init__(self, graph: WorkflowGraph):
        self.graph = graph
        self.outputs: Dict[str, Dict[str, Any]] = {}

    def get_inputs_for_node(self, node_id: str) -> Dict[str, Any]:
        inputs: Dict[str, Any] = {}
        for edge in self.graph.edges:
            if edge.dst == node_id:
                node_outputs = self.outputs.get(edge.src, {})
                if edge.src_port in node_outputs:
                    inputs[edge.dst_port] = node_outputs[edge.src_port]
        return inputs

    def set_outputs_for_node(self, node_id: str, values: Dict[str, Any]) -> None:
        self.outputs[node_id] = values


def _merge_input_payload(inputs: Dict[str, Any]) -> Dict[str, Any]:
    payload = inputs.get("data") if isinstance(inputs.get("data"), dict) else {}
    merged: Dict[str, Any] = {}
    if isinstance(payload, dict):
        merged.update(payload)
    for key in ("prompt", "negative", "text"):
        if key in inputs and key not in merged:
            merged[key] = inputs.get(key)
    return merged


def register_node_handler(node_type: str):
    def decorator(func):
        NODE_HANDLERS[node_type] = func
        return func
    return decorator


def _debug(msg: str) -> None:
    pass


def parse_graph_from_json(graph_json: str) -> WorkflowGraph:
    data = json.loads(graph_json)
    nodes_dict: Dict[str, WorkflowNode] = {}
    edges_list: List[WorkflowEdge] = []
    nodes_data = data.get("nodes")
    links_data = data.get("links")
    if isinstance(nodes_data, list) and isinstance(links_data, list):
        port_maps: Dict[str, Dict[str, List[str]]] = {}
        for n in nodes_data:
            node_id = str(n.get("id"))
            node_type = str(n.get("type"))
            config = n.get("properties") or {}
            nodes_dict[node_id] = WorkflowNode(id=node_id, type=node_type, config=config)
            inputs_meta = n.get("inputs") or []
            outputs_meta = n.get("outputs") or []
            input_names = []
            output_names = []
            for idx, meta in enumerate(inputs_meta):
                name = meta.get("name") if isinstance(meta, dict) else None
                input_names.append(name or f"in_{idx}")
            for idx, meta in enumerate(outputs_meta):
                name = meta.get("name") if isinstance(meta, dict) else None
                output_names.append(name or f"out_{idx}")
            port_maps[node_id] = {"inputs": input_names, "outputs": output_names}
        for link in links_data:
            if not isinstance(link, list) or len(link) < 6:
                continue
            src_id = str(link[1])
            src_index = int(link[2])
            dst_id = str(link[3])
            dst_index = int(link[4])
            src_ports = port_maps.get(src_id, {}).get("outputs", [])
            dst_ports = port_maps.get(dst_id, {}).get("inputs", [])
            src_port = src_ports[src_index] if 0 <= src_index < len(src_ports) else f"out_{src_index}"
            dst_port = dst_ports[dst_index] if 0 <= dst_index < len(dst_ports) else f"in_{dst_index}"
            edges_list.append(WorkflowEdge(src=src_id, src_port=src_port, dst=dst_id, dst_port=dst_port))
    else:
        nodes = data.get("nodes", {})
        edges = data.get("edges", [])
        for node_id, node_data in nodes.items():
            nodes_dict[str(node_id)] = WorkflowNode(
                id=str(node_id),
                type=str(node_data.get("type")),
                config=node_data.get("config") or {},
            )
        for e in edges:
            edges_list.append(
                WorkflowEdge(
                    src=str(e.get("src")),
                    src_port=str(e.get("src_port")),
                    dst=str(e.get("dst")),
                    dst_port=str(e.get("dst_port")),
                )
            )
    return WorkflowGraph(nodes=nodes_dict, edges=edges_list)


def topological_sort(graph: WorkflowGraph) -> List[str]:
    indegree: Dict[str, int] = {nid: 0 for nid in graph.nodes.keys()}
    for edge in graph.edges:
        indegree[edge.dst] = indegree.get(edge.dst, 0) + 1
    queue: List[str] = [nid for nid, deg in indegree.items() if deg == 0]
    order: List[str] = []
    while queue:
        nid = queue.pop(0)
        order.append(nid)
        for edge in graph.edges:
            if edge.src == nid:
                indegree[edge.dst] = indegree.get(edge.dst, 0) - 1
                if indegree[edge.dst] == 0:
                    queue.append(edge.dst)
    if len(order) != len(graph.nodes):
        return list(graph.nodes.keys())
    return order


NODE_HANDLERS: Dict[str, Any] = {}


@register_node_handler("workflow/text_input")
def handle_text_input(node: WorkflowNode, inputs: Dict[str, Any], ctx: WorkflowContext) -> Dict[str, Any]:
    prompt = node.config.get("prompt", "")
    negative = node.config.get("negative_prompt", "")
    return {"data": {"prompt": prompt, "negative": negative, "text": prompt}}


NODE_HANDLERS["workflow/input"] = handle_text_input

@register_node_handler("workflow/image_input")
def handle_image_input(node: WorkflowNode, inputs: Dict[str, Any], ctx: WorkflowContext) -> Dict[str, Any]:
    image_val = node.config.get("image")
    data = {}
    if image_val is not None:
        data["image"] = image_val
    return {"data": data}


@register_node_handler("workflow/prompt_optimizer")
def handle_prompt_optimizer(node: WorkflowNode, inputs: Dict[str, Any], ctx: WorkflowContext) -> Dict[str, Any]:
    payload = _merge_input_payload(inputs)
    base_prompt = payload.get("prompt") or node.config.get("prompt", "")
    negative_prompt = payload.get("negative") or node.config.get("negative_prompt", "")
    optimized_prompt = base_prompt
    optimized_negative = negative_prompt

    _debug(
        f"[PromptOptimizer] node_id={node.id} "
        f"input_prompt={base_prompt!r} input_negative={negative_prompt!r}"
    )

    style_name = (node.config.get("style") or "").strip()

    if prompt_optimizer_llm is not None:
        optimize_plain_fn = getattr(prompt_optimizer_llm, "optimize_prompt_llm", None)
        optimize_with_style_fn = getattr(prompt_optimizer_llm, "optimize_prompt_with_style", None)
        generate_negative_fn = getattr(prompt_optimizer_llm, "generate_negative_prompt", None)

        try:
            opt_val = None
            if style_name and style_name != "None" and callable(optimize_with_style_fn):
                _debug(f"[PromptOptimizer] node_id={node.id} use style={style_name!r}")
                opt_val = optimize_with_style_fn(base_prompt, style_name)
            elif callable(optimize_plain_fn):
                opt_val = optimize_plain_fn(base_prompt)

            if isinstance(opt_val, str) and opt_val.strip():
                optimized_prompt = opt_val
        except Exception as exc:
            _debug(f"[PromptOptimizer] node_id={node.id} optimize failed: {exc}")
            optimized_prompt = base_prompt

        if callable(generate_negative_fn):
            try:
                neg_val = generate_negative_fn(base_prompt)
                if isinstance(neg_val, str) and neg_val.strip():
                    optimized_negative = neg_val
            except Exception as exc:
                _debug(f"[PromptOptimizer] node_id={node.id} generate_negative failed: {exc}")
                optimized_negative = negative_prompt

    if not optimized_negative:
        default_neg = getattr(prompt_optimizer_llm, "DEFAULT_NEGATIVE", None) if prompt_optimizer_llm else None
        optimized_negative = default_neg or (
            "low quality, blurry, distortion, bad anatomy, extra limbs or fingers, "
            "wrong perspective, exposure problems, noise, text, watermark, logo"
        )

    _debug(
        f"[PromptOptimizer] node_id={node.id} "
        f"output_prompt={optimized_prompt!r} output_negative={optimized_negative!r}"
    )

    return {
        "data": {
            "prompt": optimized_prompt,
            "negative": optimized_negative,
            "text": optimized_prompt + "\n" + optimized_negative,
        }
    }



def _safe_int(val: Any, default: int) -> int:
    try:
        if val is None:
            return default
        return int(val)
    except Exception:
        return default

def _safe_float(val: Any, default: float) -> float:
    try:
        if val is None:
            return default
        return float(val)
    except Exception:
        return default


def _create_txt2img_processing_for_workflow(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
):
    width = max(64, _safe_int(width, 512))
    height = max(64, _safe_int(height, 512))
    steps = max(1, _safe_int(steps, 20))

    width = width - (width % 8)
    height = height - (height % 8)

    cfg_scale = getattr(opts, "cfg_scale", 7.0)

    _debug(
        f"[Txt2Img] create processing: prompt={prompt!r}, negative={negative_prompt!r}, "
        f"width={width}, height={height}, steps={steps}, cfg={cfg_scale}"
    )

    p = processing.StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=prompt,
        styles=[],
        negative_prompt=negative_prompt,
        batch_size=1,
        n_iter=1,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
    )

    p.steps = steps
    p.user = "workflow"

    return p


def _run_txt2img_processing(p):
    with closing(p):
        processed = processing.process_images(p)

    shared.total_tqdm.clear()
    return processed

def _load_image_from_payload(payload: Dict[str, Any]):
    if not isinstance(payload, dict):
        return None

    img_val = payload.get("image")
    if isinstance(img_val, str):
        if img_val.startswith("data:"):
            try:
                header, b64data = img_val.split(",", 1)
                raw = base64.b64decode(b64data)
                img = Image.open(io.BytesIO(raw))
                return images.fix_image(img)
            except Exception as exc:
                _debug(f"[Img2Img] decode dataURL failed: {exc}")
        else:
            try:
                return images.fix_image(images.read(img_val))
            except Exception as exc:
                _debug(f"[Img2Img] read image path failed: {exc}")

    paths = payload.get("images")
    if isinstance(paths, list) and paths:
        path = paths[0]
        try:
            return images.fix_image(images.read(path))
        except Exception as exc:
            _debug(f"[Img2Img] read images[0] failed: {exc}")

    prev = payload.get("preview_image")
    if isinstance(prev, str) and prev:
        path = prev
        if path.startswith("file="):
            path = path[len("file=") :]
        elif path.startswith("/file="):
            path = path[len("/file=") :]
        try:
            return images.fix_image(images.read(path))
        except Exception as exc:
            _debug(f"[Img2Img] read preview_image failed: {exc}")

    return None

@register_node_handler("workflow/txt2img")
def handle_txt2img(node: WorkflowNode, inputs: Dict[str, Any], ctx: WorkflowContext) -> Dict[str, Any]:
    payload = _merge_input_payload(inputs)
    prompt = payload.get("prompt") or node.config.get("prompt", "") or ""
    negative = payload.get("negative") or node.config.get("negative_prompt", "") or ""

    def pick_int(name: str, default: int) -> int:
        if name in node.config:
            return _safe_int(node.config.get(name), default)
        if name in payload:
            return _safe_int(payload.get(name), default)
        return default

    width = pick_int("width", 512)
    height = pick_int("height", 512)
    steps = pick_int("steps", 20)

    _debug(
        f"[Txt2Img] node_id={node.id} "
        f"prompt={prompt!r} negative={negative!r} width={width} height={height} steps={steps}"
    )

    try:
        p = _create_txt2img_processing_for_workflow(prompt, negative, width, height, steps)
        processed = _run_txt2img_processing(p)
    except Exception as exc:
        _debug(f"[Txt2Img] node_id={node.id} failed with error: {exc}")
        return {"data": {"error": f"txt2img failed: {exc}"}}

    images = getattr(processed, "images", None) or []
    _debug(f"[Txt2Img] node_id={node.id} generated {len(images)} image(s)")

    preview_url = ""
    images_info: List[str] = []

    if images:
        first = images[0]
        path = getattr(first, "already_saved_as", None)
        if isinstance(path, str) and path:
            if path.startswith("file=") or path.startswith("/file="):
                preview_url = path
            else:
                preview_url = f"file={path}"
            images_info.append(path)

    info_text = getattr(processed, "info", "") or prompt

    return {
        "data": {
            "prompt": prompt,
            "negative": negative,
            "text": info_text,
            "images": images_info,
            "preview_image": preview_url,
        }
    }

@register_node_handler("workflow/img2img")
def handle_img2img(node: WorkflowNode, inputs: Dict[str, Any], ctx: WorkflowContext) -> Dict[str, Any]:
    text_payload = inputs.get("text_data") or {}
    image_payload = inputs.get("image_data") or {}

    if not isinstance(text_payload, dict) or not isinstance(image_payload, dict):
        return {"data": {"error": "img2img requires both text_data and image_data inputs"}}

    prompt = text_payload.get("prompt") or node.config.get("prompt", "") or ""
    negative = text_payload.get("negative") or node.config.get("negative_prompt", "") or ""


    init_image = _load_image_from_payload(image_payload)
    if init_image is None:
        return {"data": {"error": "img2img failed: no valid input image"}}

    def pick_int(name: str, default: int) -> int:
        if name in node.config:
            return _safe_int(node.config.get(name), default)
        if name in text_payload:
            return _safe_int(text_payload.get(name), default)
        if name in image_payload:
            return _safe_int(image_payload.get(name), default)
        return default

    width_default = getattr(init_image, "width", 512) or 512
    height_default = getattr(init_image, "height", 512) or 512

    width = pick_int("width", width_default)
    height = pick_int("height", height_default)
    steps = pick_int("steps", 20)

    width = max(64, width)
    height = max(64, height)
    steps = max(1, steps)
    width = width - (width % 8)
    height = height - (height % 8)

    cfg_scale = getattr(opts, "cfg_scale", 7.0)

    raw_denoise = None
    if "denoising_strength" in node.config:
        raw_denoise = node.config.get("denoising_strength")
    elif "denoising_strength" in text_payload:
        raw_denoise = text_payload.get("denoising_strength")
    elif "denoising_strength" in image_payload:
        raw_denoise = image_payload.get("denoising_strength")

    default_denoise = getattr(opts, "img2img_denoising_strength", 0.6)
    denoising = _safe_float(raw_denoise, default_denoise)

    if denoising < 0.0:
        denoising = 0.0
    if denoising > 1.0:
        denoising = 1.0

    _debug(
        f"[Img2Img] node_id={node.id} "
        f"prompt={prompt!r} negative={negative!r} width={width} height={height} "
        f"steps={steps} denoising={denoising} cfg={cfg_scale}"
    )

    try:
        p = processing.StableDiffusionProcessingImg2Img(
            sd_model=shared.sd_model,
            outpath_samples=opts.outdir_samples or opts.outdir_img2img_samples,
            outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
            prompt=prompt,
            negative_prompt=negative,
            styles=[],
            batch_size=1,
            n_iter=1,
            cfg_scale=cfg_scale,
            width=width,
            height=height,
            init_images=[init_image],
            denoising_strength=denoising,
        )

        p.steps = steps
        p.user = "workflow"

        processed = _run_txt2img_processing(p)

    except Exception as exc:
        _debug(f"[Img2Img] node_id={node.id} failed with error: {exc}")
        return {"data": {"error": f"img2img failed: {exc}"}}

    images_list = getattr(processed, "images", None) or []
    _debug(f"[Img2Img] node_id={node.id} generated {len(images_list)} image(s)")

    preview_url = ""
    images_info: List[str] = []

    if images_list:
        first = images_list[0]
        path = getattr(first, "already_saved_as", None)
        if isinstance(path, str) and path:
            if path.startswith("file=") or path.startswith("/file="):
                preview_url = path
            else:
                preview_url = f"file={path}"
            images_info.append(path)

    info_text = getattr(processed, "info", "") or prompt

    return {
        "data": {
            "prompt": prompt,
            "negative": negative,
            "text": info_text,
            "images": images_info,
            "preview_image": preview_url,
        }
    }

@register_node_handler("workflow/output")
def handle_output(node: WorkflowNode, inputs: Dict[str, Any], ctx: WorkflowContext) -> Dict[str, Any]:
    payload = _merge_input_payload(inputs)

    error_msg = payload.get("error")
    if error_msg:
        return {
            "data": {
                "text": f"[Error] {error_msg}",
                "prompt": "",
                "negative": "",
                "images": None,
                "preview_image": None,
            }
        }

    prompt_val = payload.get("prompt", "") or ""
    negative_val = payload.get("negative", "") or ""
    text_val = payload.get("text") or prompt_val
    if negative_val:
        text_val = f"{prompt_val}\n{negative_val}"

    images = payload.get("images")
    preview_image = payload.get("preview_image")

    return {
        "data": {
            "text": text_val,
            "prompt": prompt_val,
            "negative": negative_val,
            "images": images,
            "preview_image": preview_image,
        }
    }


NODE_HANDLERS["workflow/output_text"] = handle_output
NODE_HANDLERS["workflow/output_image"] = handle_output


def execute_workflow(graph: WorkflowGraph) -> WorkflowContext:
    ctx = WorkflowContext(graph)
    order = topological_sort(graph)
    for node_id in order:
        node = graph.nodes.get(node_id)
        if node is None:
            continue
        handler = NODE_HANDLERS.get(node.type)
        if handler is None:
            ctx.set_outputs_for_node(node_id, {})
            continue
        inputs = ctx.get_inputs_for_node(node_id)
        try:
            _debug(f"Executing node {node.type} ({node_id}) with inputs keys: {list(inputs.keys())}")
            outputs = handler(node, inputs, ctx)
            _debug(f"Node {node.type} ({node_id}) produced keys: {list((outputs or {}).keys())}")
        except Exception as exc:
            _debug(f"Node {node.type} ({node_id}) failed: {exc}")
            outputs = {"data": {"error": str(exc)}}
        ctx.set_outputs_for_node(node_id, outputs or {})
    return ctx


def build_node_outputs_payload(ctx: WorkflowContext) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for nid, outputs in ctx.outputs.items():
        summary: Dict[str, Any] = {}
        data = outputs.get("data", outputs)
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    summary[key] = value
        payload[nid] = summary
    return payload


def run_workflow(graph_json: str):
    if not graph_json:
        return "", json.dumps({})
    _debug(f"run_workflow invoked, graph_json length={len(graph_json)}")
    graph = parse_graph_from_json(graph_json)
    _debug(f"Parsed graph nodes={len(graph.nodes)} edges={len(graph.edges)}")
    ctx = execute_workflow(graph)
    _debug(f"Collected outputs for nodes: {list(ctx.outputs.keys())}")
    out_text = ""
    for nid, node in graph.nodes.items():
        if node.type in ("workflow/output", "workflow/output_text"):
            data = ctx.outputs.get(nid, {}).get("data", {})
            if isinstance(data, dict) and "text" in data:
                out_text = str(data.get("text") or "")
                break
    if not out_text and ctx.outputs:
        for outputs in ctx.outputs.values():
            data = outputs.get("data", outputs)
            if isinstance(data, dict) and "text" in data:
                out_text = str(data.get("text") or "")
                break
    node_outputs = build_node_outputs_payload(ctx)
    _debug(f"Returning text length={len(out_text)} node_outputs_keys={list(node_outputs.keys())}")
    return out_text, json.dumps(node_outputs, ensure_ascii=False)
