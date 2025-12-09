import json
from dataclasses import dataclass, field
from typing import Any, Dict, List
from contextlib import closing

from modules import processing
import modules.shared as shared
from modules.shared import opts

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
    """
    把上游节点通过端口传过来的 data 合并成一个扁平 payload：
    - 先拿 inputs["data"]（如果是 dict）
    - 再补充 prompt / negative / text 顶层字段
    其他字段（例如 images / preview_image / width / height / steps 等）会自然保留下来。
    """
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
    """统一的调试输出入口，方便随时关/开。"""
    try:
        print(f"[Workflow] {msg}", flush=True)
    except Exception:
        pass


def parse_graph_from_json(graph_json: str) -> WorkflowGraph:
    data = json.loads(graph_json)
    nodes_dict: Dict[str, WorkflowNode] = {}
    edges_list: List[WorkflowEdge] = []
    nodes_data = data.get("nodes")
    links_data = data.get("links")
    if isinstance(nodes_data, list) and isinstance(links_data, list):
        # LiteGraph 风格
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
        # 简单自定义格式
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
        # 有环就退化为任意顺序
        return list(graph.nodes.keys())
    return order


NODE_HANDLERS: Dict[str, Any] = {}


@register_node_handler("workflow/input")
def handle_input(node: WorkflowNode, inputs: Dict[str, Any], ctx: WorkflowContext) -> Dict[str, Any]:
    prompt = node.config.get("prompt", "")
    negative = node.config.get("negative_prompt", "")
    return {"data": {"prompt": prompt, "negative": negative, "text": prompt}}


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

    styles = node.config.get("style")
    if styles and hasattr(prompt_optimizer_llm, "DEFAULT_NEGATIVE"):
        # style 先留着做扩展
        pass

    if prompt_optimizer_llm is not None:
        optimize_fn = getattr(prompt_optimizer_llm, "optimize_prompt_llm", None)
        generate_negative_fn = getattr(prompt_optimizer_llm, "generate_negative_prompt", None)
        if callable(optimize_fn):
            try:
                opt_val = optimize_fn(base_prompt)
                if isinstance(opt_val, str) and opt_val.strip():
                    optimized_prompt = opt_val
            except Exception:
                optimized_prompt = base_prompt
        if callable(generate_negative_fn):
            try:
                neg_val = generate_negative_fn(base_prompt)
                if isinstance(neg_val, str) and neg_val.strip():
                    optimized_negative = neg_val
            except Exception:
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


def _create_txt2img_processing_for_workflow(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
):
    """
    极简版 txt2img：
    - 不经过 modules.txt2img / scripts 系统，避免脚本参数数量不匹配
    - 直接构造 StableDiffusionProcessingTxt2Img，然后 process_images
    """
    width = max(64, _safe_int(width, 512))
    height = max(64, _safe_int(height, 512))
    steps = max(1, _safe_int(steps, 20))

    # 8 的倍数对齐
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

    # 手动设置步数
    p.steps = steps
    # 区分一下来源
    p.user = "workflow"

    return p


def _run_txt2img_processing(p):
    """
    极简执行：直接 process_images，不走 scripts_txt2img.run
    """
    with closing(p):
        processed = processing.process_images(p)

    shared.total_tqdm.clear()
    return processed


@register_node_handler("workflow/txt2img")
def handle_txt2img(node: WorkflowNode, inputs: Dict[str, Any], ctx: WorkflowContext) -> Dict[str, Any]:
    """
    文生图节点：
    - 输入：从上游拿 prompt / negative（可以来自 Input 或 Prompt Optimizer）
    - 节点属性：width / height / steps
    - 输出：images（路径列表，给后续节点用）、preview_image（字符串 URL，给前端 Output 预览用）、text/info
    """
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


@register_node_handler("workflow/output")
def handle_output(node: WorkflowNode, inputs: Dict[str, Any], ctx: WorkflowContext) -> Dict[str, Any]:
    """
    工作流终点：
    - 兼容纯文本链路（prompt/negative/text）
    - 也兼容图片链路（images/preview_image），原样透传，方便前端 Output inspector 判断展示模式。
    - 如果上游有 error 字段，则直接在文本里显示错误。
    """
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
    """
    将每个节点的输出压缩成只包含基础类型的 summary，
    用于写回前端的 workflow_node_outputs_json。
    """
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
    # collect terminal output text
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
