# modules/ui_prompt_optimizer.py

import gradio as gr
from modules import prompt_optimizer_llm


def create_ui():

    gr.Markdown("## Prompt Optimizer（本地 LLM）")

    with gr.Row():
        with gr.Column(scale=1):
            input_box = gr.Textbox(
                label="原始 Prompt",
                lines=3,
                placeholder="例如：cyberpunk girl neon city",
            )

            with gr.Row():
                btn_optimize = gr.Button("一键优化 Prompt")
                btn_negative = gr.Button("生成 Negative Prompt")

        with gr.Column(scale=1):
            output_box = gr.Textbox(
                label="优化后 Prompt（专业英文）",
                lines=4,
            )
            negative_box = gr.Textbox(
                label="自动生成 Negative Prompt",
                lines=4,
                placeholder="根据原始 Prompt 自动生成 Negative Prompt",
            )

    def _optimize(prompt: str) -> str:
        if not prompt.strip():
            return ""
        return prompt_optimizer_llm.optimize_prompt_llm(prompt)

    def _make_negative(prompt: str) -> str:
        if not prompt.strip():
            return ""
        return prompt_optimizer_llm.generate_negative_prompt(prompt)

    btn_optimize.click(
        fn=_optimize,
        inputs=input_box,
        outputs=output_box,
    )

    btn_negative.click(
        fn=_make_negative,
        inputs=input_box,
        outputs=negative_box,
    )
