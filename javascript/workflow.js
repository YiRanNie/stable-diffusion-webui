// Workflow UI with Input / Prompt Optimizer / Txt2Img / Text Output / Image Output

function workflowGetApp() {
    if (window.gradioApp) return window.gradioApp();
    return document.querySelector("body");
}

function workflowFindElement(selector) {
    const app = workflowGetApp();
    return app ? app.querySelector(selector) : null;
}

function workflowFindComponent(elemId, innerSelector) {
    const app = workflowGetApp();
    if (!app) return null;
    const container = app.querySelector(`#${elemId}`);
    if (!container) return null;
    if (innerSelector) {
        const inner = container.querySelector(innerSelector);
        return inner || container;
    }
    return container;
}

function workflowSerializeGraph(graph) {
    return JSON.stringify(graph.serialize());
}

function workflowSyncGraph(graph) {
    const textbox = workflowFindComponent("workflow_graph_json", "textarea");
    if (!textbox) {
        setTimeout(() => workflowSyncGraph(graph), 300);
        return;
    }
    textbox.value = workflowSerializeGraph(graph);
    textbox.dispatchEvent(new Event("input", { bubbles: true }));
}

// === node outputs 同步 & 去抖 ===

let workflowLastNodeOutputsRaw = null;
window.workflowNodeOutputs = window.workflowNodeOutputs || {};

function workflowSyncNodeOutputs() {
    const textbox = workflowFindComponent("workflow_node_outputs_json", "textarea");
    if (!textbox) return;

    const raw = textbox.value || "{}";

    // 如果和上次完全一样，说明后端没有新结果，无需重新渲染
    if (raw === workflowLastNodeOutputsRaw) {
        return;
    }
    workflowLastNodeOutputsRaw = raw;

    let parsed = {};
    try {
        parsed = JSON.parse(raw);
    } catch (e) {
        parsed = {};
    }
    window.workflowNodeOutputs = parsed;

    // 有非空结果就停表，并固定显示时间
    if (parsed && Object.keys(parsed).length > 0) {
        workflowStopTimer(true);
    }

    const render = window._workflowRenderInspector;
    const currentNode = window._workflowCurrentNode;
    if (
        render &&
        currentNode &&
        (currentNode.type === "workflow/output" ||
            currentNode.type === "workflow/output_text" ||
            currentNode.type === "workflow/output_image")
    ) {
        render(currentNode);
    }
}

function workflowBindNodeOutputsListener(allowRetry = true) {
    const textbox = workflowFindComponent("workflow_node_outputs_json", "textarea");
    if (!textbox) {
        if (allowRetry) setTimeout(() => workflowBindNodeOutputsListener(false), 400);
        return;
    }
    if (textbox._workflowBound) return;
    textbox._workflowBound = true;

    const sync = () => workflowSyncNodeOutputs();

    textbox.addEventListener("input", sync);
    textbox.addEventListener("change", sync);

    if (window.MutationObserver) {
        const observer = new MutationObserver(sync);
        observer.observe(textbox, { characterData: true, subtree: true, childList: true });
    }

    if (!textbox._workflowSyncTimer) {
        textbox._workflowSyncTimer = setInterval(sync, 1000);
    }

    sync();
}

// === 节点 schema 定义 ===

const WORKFLOW_NODE_SCHEMAS = {
    "workflow/input": {
        title: "Input",
        inputs: [],
        outputs: ["data"],
        defaults: {
            prompt: "",
            negative_prompt: "",
        },
        fields: [
            { key: "prompt", label: "Prompt", type: "textarea", placeholder: "Enter prompt", rows: 8 },
            { key: "negative_prompt", label: "Negative Prompt", type: "textarea", placeholder: "Enter negative prompt", rows: 8 },
        ],
    },
    "workflow/prompt_optimizer": {
        title: "Prompt Optimizer",
        inputs: ["data"],
        outputs: ["data"],
        defaults: {},
        fields: [],
    },
    "workflow/txt2img": {
        title: "Txt2Img",
        inputs: ["data"],
        outputs: ["data"],
        defaults: {
            width: 512,
            height: 512,
            steps: 20,
        },
        fields: [
            { key: "width", label: "Width", type: "number", min: 64, step: 8 },
            { key: "height", label: "Height", type: "number", min: 64, step: 8 },
            { key: "steps", label: "Sampling Steps", type: "number", min: 1, step: 1 },
        ],
    },
    "workflow/output_text": {
        title: "Text Output",
        inputs: ["data"],
        outputs: [],
        defaults: {},
        fields: [],
    },
    "workflow/output_image": {
        title: "Image Output",
        inputs: ["data"],
        outputs: [],
        defaults: {},
        fields: [],
    },
    // 老的 workflow/output 仍然支持，防止已有图崩掉（palette 不再展示）
    "workflow/output": {
        title: "Output",
        inputs: ["data"],
        outputs: [],
        defaults: {},
        fields: [],
    },
};

// === 计时器状态 ===

let workflowTimerId = null;
let workflowTimerStart = null;
window._workflowTimerBox = window._workflowTimerBox || null;

function workflowStartTimer() {
    const box = window._workflowTimerBox;
    if (!box) return;

    workflowLastNodeOutputsRaw = null;
    window.workflowNodeOutputs = {};

    workflowStopTimer(false);
    workflowTimerStart = performance.now();
    box.textContent = "0.0s";
    box.classList.add("workflow-timer-running");

    workflowTimerId = setInterval(() => {
        if (!workflowTimerStart) return;
        const elapsed = (performance.now() - workflowTimerStart) / 1000;
        box.textContent = elapsed.toFixed(1) + "s";
    }, 100);
}

function workflowStopTimer(finalize = true) {
    if (workflowTimerId !== null) {
        clearInterval(workflowTimerId);
        workflowTimerId = null;
    }
    const box = window._workflowTimerBox;
    if (!box) {
        workflowTimerStart = null;
        return;
    }
    if (workflowTimerStart && finalize) {
        const elapsed = (performance.now() - workflowTimerStart) / 1000;
        box.textContent = elapsed.toFixed(1) + "s";
    }
    workflowTimerStart = null;
    box.classList.remove("workflow-timer-running");
}

// === palette node ===

function workflowCreatePaletteNode(type) {
    const meta = WORKFLOW_NODE_SCHEMAS[type] || {};
    const item = document.createElement("div");
    item.className = "workflow-palette-item";
    item.textContent = meta.title || type;
    item.dataset.nodeType = type;
    item.draggable = true;
    item.addEventListener("dragstart", (e) => {
        e.dataTransfer.effectAllowed = "copy";
        e.dataTransfer.setData("text/plain", type);
    });
    return item;
}

// === 样式注入 ===

function workflowInjectStyle() {
    if (document.getElementById("workflow-style")) return;
    const style = document.createElement("style");
    style.id = "workflow-style";
    style.textContent = `
    #workflow_canvas_root .workflow-root {
        display: grid;
        grid-template-columns: 220px minmax(400px, 1fr) 280px;
        grid-template-rows: minmax(520px, 1fr);
        gap: 10px;
        height: 680px;
        box-sizing: border-box;
    }
    #workflow_canvas_root .workflow-palette {
        background: #0e121a;
        border-radius: 10px;
        border: 1px solid #2b3340;
        padding: 10px;
        display: flex;
        flex-direction: column;
        overflow-y: auto;
    }
    #workflow_canvas_root .workflow-palette-item {
        font-size: 12px;
        padding: 7px 9px;
        border-radius: 8px;
        margin-bottom: 6px;
        background: #151924;
        color: #e5e7eb;
        cursor: grab;
        border: 1px solid transparent;
    }
    #workflow_canvas_root .workflow-canvas {
        position: relative;
        background: #05070d;
        border-radius: 10px;
        border: 1px solid #2b3340;
        overflow: hidden;
    }
    #workflow_canvas_root #workflow_graph_canvas {
        width: 100%;
        height: 100%;
        display: block;
    }
    #workflow_canvas_root .workflow-inspector {
        display: flex;
        flex-direction: column;
        background: #0e121a;
        border-radius: 10px;
        border: 1px solid #2b3340;
        padding: 10px;
        box-sizing: border-box;
        height: 100%;
    }
    #workflow_canvas_root .workflow-inspector-main {
        flex: 1;
        overflow-y: auto;
        margin-bottom: 8px;
    }
    #workflow_canvas_root .workflow-inspector-field {
        display: flex;
        flex-direction: column;
        gap: 4px;
        margin-bottom: 8px;
    }
    #workflow_canvas_root .workflow-inspector-field label {
        font-size: 11px;
        color: #9ca3af;
    }
    #workflow_canvas_root .workflow-inspector-field textarea,
    #workflow_canvas_root .workflow-inspector-field input {
        background: #05070d;
        border-radius: 8px;
        border: 1px solid #374151;
        font-size: 12px;
        padding: 6px 8px;
        color: #e5e7eb;
    }
    #workflow_canvas_root .workflow-run-container {
        margin-top: 4px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
    }
    #workflow_canvas_root .workflow-run-button {
        padding: 10px 24px;
        font-size: 13px;
        border-radius: 999px;
        border: 1px solid #4f46e5;
        background: #4f46e5;
        color: #e5e7eb;
        cursor: pointer;
    }
    #workflow_canvas_root .workflow-run-button:hover {
        filter: brightness(1.05);
    }
    #workflow_canvas_root .workflow-timer {
        min-width: 64px;
        padding: 4px 8px;
        font-size: 11px;
        border-radius: 8px;
        border: 1px solid #374151;
        background: #05070d;
        color: #9ca3af;
        text-align: center;
        box-sizing: border-box;
    }
    #workflow_canvas_root .workflow-timer.workflow-timer-running {
        border-color: #4f46e5;
        color: #e5e7eb;
    }
    #workflow_canvas_root .workflow-image-preview {
        width: 100%;
        border-radius: 10px;
        border: 1px solid #374151;
        background: #05070d;
        object-fit: contain;
        max-height: 260px;
    }
    #workflow_canvas_root .workflow-download-button {
        margin-top: 6px;
        align-self: flex-start;
        padding: 4px 10px;
        font-size: 11px;
        border-radius: 999px;
        border: 1px solid #4f46e5;
        background: #4f46e5;
        color: #e5e7eb;
        cursor: pointer;
    }

    /* 全屏预览遮罩 */
    .workflow-img-overlay {
        position: fixed;
        inset: 0;
        background: rgba(0, 0, 0, 0.85);
        display: none;
        align-items: center;
        justify-content: center;
        z-index: 9999;
    }
    .workflow-img-overlay-img {
        max-width: 95vw;
        max-height: 95vh;
        border-radius: 12px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6);

    `;
    document.head.appendChild(style);
}

// === 初始化 LiteGraph ===

function workflowInitLiteGraph() {
    const root = workflowFindElement("#workflow_canvas_root");
    if (!root || root._workflowInitialized || !window.LiteGraph) return;
    root._workflowInitialized = true;

    root.innerHTML = "";
    workflowInjectStyle();

    const container = document.createElement("div");
    container.className = "workflow-root";

    const palette = document.createElement("div");
    palette.className = "workflow-palette";
    palette.appendChild(workflowCreatePaletteNode("workflow/input"));
    palette.appendChild(workflowCreatePaletteNode("workflow/prompt_optimizer"));
    palette.appendChild(workflowCreatePaletteNode("workflow/txt2img"));
    palette.appendChild(workflowCreatePaletteNode("workflow/output_text"));
    palette.appendChild(workflowCreatePaletteNode("workflow/output_image"));

    const canvasWrapper = document.createElement("div");
    canvasWrapper.className = "workflow-canvas";
    const canvas = document.createElement("canvas");
    canvas.id = "workflow_graph_canvas";
    canvas.tabIndex = 0;
    canvasWrapper.appendChild(canvas);

    const inspector = document.createElement("div");
    inspector.className = "workflow-inspector";
    const inspectorMain = document.createElement("div");
    inspectorMain.className = "workflow-inspector-main";
    inspector.appendChild(inspectorMain);

    const runContainer = document.createElement("div");
    runContainer.className = "workflow-run-container";

    const timerBox = document.createElement("div");
    timerBox.className = "workflow-timer";
    timerBox.textContent = "0.0s";
    runContainer.appendChild(timerBox);
    window._workflowTimerBox = timerBox;

    const runButton = document.createElement("button");
    runButton.className = "workflow-run-button";
    runButton.textContent = "Run Workflow";
    runButton.addEventListener("click", () => {
        workflowStartTimer();
        workflowSyncGraph(graph);
        setTimeout(() => {
            const hidden = workflowFindComponent("workflow_run_button", "button");
            if (hidden) hidden.click();
        }, 30);
    });
    runContainer.appendChild(runButton);
    inspector.appendChild(runContainer);

    container.appendChild(palette);
    container.appendChild(canvasWrapper);
    container.appendChild(inspector);
    root.appendChild(container);

    const graph = new LiteGraph.LGraph();
    const graphCanvas = new LiteGraph.LGraphCanvas("#workflow_graph_canvas", graph);
    window.workflowGraphInstance = graph;
    window.workflowGraphCanvas = graphCanvas;
    if (graphCanvas.ds) {
        graphCanvas.ds.scale = 1;
        graphCanvas.ds.offset = [0, 0];
    }

    function applyNodeDefaults(node) {
        const meta = WORKFLOW_NODE_SCHEMAS[node.type];
        if (!meta) return;
        node.properties = node.properties || {};
        Object.keys(meta.defaults || {}).forEach((k) => {
            if (node.properties[k] === undefined) node.properties[k] = meta.defaults[k];
        });
    }

    function renderField(node, field) {
        const wrap = document.createElement("div");
        wrap.className = "workflow-inspector-field";

        const label = document.createElement("label");
        label.textContent = field.label || field.key;
        wrap.appendChild(label);

        const props = node.properties || {};
        const current =
            props[field.key] !== undefined
                ? props[field.key]
                : field.default !== undefined
                ? field.default
                : "";

        let input;
        if (field.type === "textarea") {
            input = document.createElement("textarea");
            input.value = current || "";
            if (field.placeholder) input.placeholder = field.placeholder;
            input.rows = field.rows || 6;
            input.addEventListener("input", () => {
                props[field.key] = input.value;
                workflowSyncGraph(graph);
            });
        } else if (field.type === "number") {
            input = document.createElement("input");
            input.type = "number";
            input.value = current;
            if (field.min !== undefined) input.min = String(field.min);
            if (field.max !== undefined) input.max = String(field.max);
            if (field.step !== undefined) input.step = String(field.step);
            input.addEventListener("input", () => {
                const v = input.value;
                props[field.key] = v === "" ? "" : Number(v);
                workflowSyncGraph(graph);
            });
        } else {
            input = document.createElement("input");
            input.type = "text";
            input.value = current || "";
            if (field.placeholder) input.placeholder = field.placeholder;
            input.addEventListener("input", () => {
                props[field.key] = input.value;
                workflowSyncGraph(graph);
            });
        }
        wrap.appendChild(input);
        return wrap;
    }

    function workflowRenderInspector(node) {
        if (!inspectorMain) return;
        inspectorMain.innerHTML = "";
        if (!node) return;

        applyNodeDefaults(node);
        const meta = WORKFLOW_NODE_SCHEMAS[node.type];
        if (meta && Array.isArray(meta.fields)) {
            meta.fields.forEach((field) => {
                const el = renderField(node, field);
                if (el) inspectorMain.appendChild(el);
            });
        }

        // Text Output 节点：只显示文本
        if (node.type === "workflow/output_text" || node.type === "workflow/output") {
            const allOutputs = window.workflowNodeOutputs || {};
            const nodeOutputs = allOutputs[String(node.id)] || {};

            const wrap = document.createElement("div");
            wrap.className = "workflow-inspector-field";

            const label = document.createElement("label");
            label.textContent = "Output Text";
            wrap.appendChild(label);

            const textarea = document.createElement("textarea");
            textarea.readOnly = true;
            textarea.value = nodeOutputs.text || "";
            textarea.rows = 10;
            textarea.style.resize = "vertical";

            wrap.appendChild(textarea);
            inspectorMain.appendChild(wrap);
        }

        // Image Output 节点：只显示图片 + 下载按钮
        if (node.type === "workflow/output_image") {
            const allOutputs = window.workflowNodeOutputs || {};
            const nodeOutputs = allOutputs[String(node.id)] || {};
            const src = nodeOutputs.preview_image || "";

            const wrap = document.createElement("div");
            wrap.className = "workflow-inspector-field";

            const label = document.createElement("label");
            label.textContent = "Preview Image";
            wrap.appendChild(label);

            if (src) {
                const img = document.createElement("img");
                img.className = "workflow-image-preview";
                img.src = src;

                img.addEventListener("click", () => {
                    workflowShowImageLightbox(src);
                });

                wrap.appendChild(img);

                const btn = document.createElement("a");
                btn.className = "workflow-download-button";
                btn.textContent = "Download";
                btn.href = src;
                btn.download = "output.png";
                wrap.appendChild(btn);
            } else {
                const span = document.createElement("span");
                span.style.fontSize = "11px";
                span.style.color = "#6b7280";
                span.textContent = "No image yet.";
                wrap.appendChild(span);
            }

            inspectorMain.appendChild(wrap);
        }
    }

    window._workflowRenderInspector = workflowRenderInspector;

    function defineNode(type, meta, executor) {
        function Node() {
            this.properties = Object.assign({}, meta.defaults || {});
            (meta.inputs || []).forEach((label) => this.addInput(label, "*"));
            (meta.outputs || []).forEach((label) => this.addOutput(label, "*"));
        }
        Node.title = meta.title || type;
        Node.prototype.onExecute = executor;
        Node.prototype.onPropertyChanged = function (name, value) {
            this.properties[name] = value;
            workflowSyncGraph(graph);
            return true;
        };
        LiteGraph.registerNodeType(type, Node);
    }

    // Input
    defineNode("workflow/input", WORKFLOW_NODE_SCHEMAS["workflow/input"], function () {
        const p = this.properties || {};
        this.setOutputData(0, { prompt: p.prompt || "", negative: p.negative_prompt || "", text: p.prompt || "" });
    });

    // Prompt Optimizer（前端只转发数据，真正优化在后端）
    defineNode("workflow/prompt_optimizer", WORKFLOW_NODE_SCHEMAS["workflow/prompt_optimizer"], function () {
        const incoming = this.getInputData(0) || {};
        this.setOutputData(0, {
            prompt: incoming.prompt || "",
            negative: incoming.negative || "",
            text: incoming.text || incoming.prompt || "",
        });
    });

    // Txt2Img 节点：前端只传递参数，真正生图在后端
    defineNode("workflow/txt2img", WORKFLOW_NODE_SCHEMAS["workflow/txt2img"], function () {
        const incoming = this.getInputData(0) || {};
        const p = this.properties || {};
        this.setOutputData(0, {
            prompt: incoming.prompt || "",
            negative: incoming.negative || "",
            text: incoming.text || incoming.prompt || "",
            width: p.width,
            height: p.height,
            steps: p.steps,
        });
    });

    // Text Output：只做透传
    defineNode("workflow/output_text", WORKFLOW_NODE_SCHEMAS["workflow/output_text"], function () {
        const incoming = this.getInputData(0) || {};
        this.setOutputData(0, incoming);
    });

    // Image Output：只做透传
    defineNode("workflow/output_image", WORKFLOW_NODE_SCHEMAS["workflow/output_image"], function () {
        const incoming = this.getInputData(0) || {};
        this.setOutputData(0, incoming);
    });

    // 兼容旧的 workflow/output（和 Text Output 类似）
    defineNode("workflow/output", WORKFLOW_NODE_SCHEMAS["workflow/output"], function () {
        const incoming = this.getInputData(0) || {};
        this.setOutputData(0, incoming);
    });

    graph.onAfterChange = () => workflowSyncGraph(graph);
    graphCanvas.onNodeSelected = (node) => {
        window._workflowCurrentNode = node;
        workflowRenderInspector(node);
    };
    graphCanvas.onNodeDeselected = () => {
        window._workflowCurrentNode = null;
        workflowRenderInspector(null);
    };

    function resizeCanvas() {
        const rect = canvasWrapper.getBoundingClientRect();
        if (!rect.width || !rect.height) return;
        canvas.width = rect.width;
        canvas.height = rect.height;
        graphCanvas.resize(canvas.width, canvas.height);
    }
    if (window.ResizeObserver) {
        new ResizeObserver(() => resizeCanvas()).observe(canvasWrapper);
    }
    window.addEventListener("resize", resizeCanvas);
    setTimeout(resizeCanvas, 50);

    canvas.addEventListener("dragover", (e) => e.preventDefault());
    canvas.addEventListener("drop", (e) => {
        e.preventDefault();
        const type = (e.dataTransfer && e.dataTransfer.getData("text/plain")) || "";
        if (!type) return;
        const pos = graphCanvas.convertEventToCanvasOffset(e);
        const node = LiteGraph.createNode(type);
        if (!node) return;
        applyNodeDefaults(node);
        node.pos = pos || [80, 80];
        graph.add(node);
        workflowSyncGraph(graph);
        workflowRenderInspector(node);
    });

    workflowSyncGraph(graph);

    graph.start();
    workflowBindNodeOutputsListener();
}

// === 加载 LiteGraph ===

function workflowEnsureLiteGraphLoaded() {
    if (window.LiteGraph) {
        workflowInitLiteGraph();
        return;
    }
    const script = document.createElement("script");
    script.src = "https://unpkg.com/litegraph.js/build/litegraph.min.js";
    script.async = true;
    script.dataset.workflowLitegraph = "1";
    script.onload = () => workflowInitLiteGraph();
    document.head.appendChild(script);
}

function initWorkflowUI() {
    const check = () => {
        const root = workflowFindElement("#workflow_canvas_root");
        if (!root) {
            setTimeout(check, 400);
            return;
        }
        workflowEnsureLiteGraphLoaded();
    };
    check();
    if (window.onUiTabChange && !window._workflowTabHooked) {
        window._workflowTabHooked = true;
        onUiTabChange((name) => {
            if (name === "workflow") workflowEnsureLiteGraphLoaded();
        });
    }
}

if (window.onUiLoaded) {
    onUiLoaded(initWorkflowUI);
} else {
    window.addEventListener("load", initWorkflowUI);
}

// === 全屏预览辅助 ===
function workflowShowImageLightbox(src) {
    if (!src) return;

    let overlay = document.getElementById("workflow-img-overlay");
    if (!overlay) {
        overlay = document.createElement("div");
        overlay.id = "workflow-img-overlay";
        overlay.className = "workflow-img-overlay";

        const img = document.createElement("img");
        img.className = "workflow-img-overlay-img";
        overlay.appendChild(img);

        // 点击任意位置关闭
        overlay.addEventListener("click", () => {
            workflowHideImageLightbox();
        });

        document.body.appendChild(overlay);
    }

    const img = overlay.querySelector("img");
    img.src = src;
    overlay.style.display = "flex";
}

function workflowHideImageLightbox() {
    const overlay = document.getElementById("workflow-img-overlay");
    if (overlay) {
        overlay.style.display = "none";
    }
}
