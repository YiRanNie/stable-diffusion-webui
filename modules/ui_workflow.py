import gradio as gr

from modules import workflow_engine


def create_workflow_ui():
    with gr.Column(elem_id="workflow_root"):
        graph_json = gr.Textbox(
            label="Workflow Graph JSON",
            visible=False,
            elem_id="workflow_graph_json",
        )
        node_outputs_json = gr.Textbox(
            label="Workflow Node Outputs",
            visible=False,
            elem_id="workflow_node_outputs_json",
        )
        with gr.Row():
            gr.HTML(
                value="<div id='workflow_canvas_root' style='width:100%;height:700px;'></div>",
                elem_id="workflow_canvas_root_container",
            )
        run_button = gr.Button(
            value="Run Workflow",
            elem_id="workflow_run_button",
            visible=False,
        )
        # 仍然接收 out_text，但不对用户显示
        text_output = gr.Textbox(
            label="Text Output",
            elem_id="workflow_output_text",
            lines=6,
            interactive=False,
            visible=False,  # 隐藏底部输出
        )

        run_button.click(
            fn=workflow_engine.run_workflow,
            inputs=[graph_json],
            outputs=[text_output, node_outputs_json],
        )
