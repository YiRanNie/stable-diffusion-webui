import gradio as gr
from modules import prompt_optimizer_llm


def create_ui():

    gr.Markdown("## Prompt Optimizer")

    with gr.Row():
        with gr.Column(scale=1):
            input_box = gr.Textbox(
                label="Original Prompt",
                lines=3,
                placeholder="e.g. cyberpunk girl neon city",
            )

            style_dropdown = gr.Dropdown(
                label="Style",
                choices=[
                    "None",
                    "Photorealistic",
                    "Anime",
                    "Illustration",
                    "Cinematic",
                    "3D Render",
                ],
                value="None",
            )

            with gr.Row():
                btn_optimize = gr.Button("Optimize Prompt")
                btn_negative = gr.Button("Generate Negative Prompt")

        with gr.Column(scale=1):
            output_box = gr.Textbox(
                label="Optimized Prompt",
                lines=4,
                elem_id="prompt_optimizer_optimized_box",
            )
            btn_copy_output = gr.Button(
                "Copy Optimized Prompt",
                elem_id="prompt_optimizer_copy_optimized_btn",
            )

            negative_box = gr.Textbox(
                label="Negative Prompt",
                lines=4,
                placeholder="",
                elem_id="prompt_optimizer_negative_box",
            )
            btn_copy_negative = gr.Button(
                "Copy Negative Prompt",
                elem_id="prompt_optimizer_copy_negative_btn",
            )

    def _optimize(prompt: str, style: str) -> str:
        if not prompt.strip():
            return ""
        return prompt_optimizer_llm.optimize_prompt_with_style(prompt, style)

    def _make_negative(prompt: str) -> str:
        if not prompt.strip():
            return ""
        return prompt_optimizer_llm.generate_negative_prompt(prompt)

    btn_optimize.click(
        fn=_optimize,
        inputs=[input_box, style_dropdown],
        outputs=output_box,
    )

    btn_negative.click(
        fn=_make_negative,
        inputs=input_box,
        outputs=negative_box,
    )

    btn_copy_output.click(
        None,
        None,
        None,
        _js="""
        function(){
            const app = gradioApp();
            const box = app.querySelector('#prompt_optimizer_optimized_box textarea');
            const btn = app.querySelector('#prompt_optimizer_copy_optimized_btn');
            if (!box || !btn) return;

            navigator.clipboard.writeText(box.value || "");

            const span = btn.querySelector('span');
            const target = span || btn;
            target.innerText = "Copied";
            setTimeout(() => { target.innerText = "Copy Optimized Prompt"; }, 1200);
        }
        """
    )

    btn_copy_negative.click(
        None,
        None,
        None,
        _js="""
        function(){
            const app = gradioApp();
            const box = app.querySelector('#prompt_optimizer_negative_box textarea');
            const btn = app.querySelector('#prompt_optimizer_copy_negative_btn');
            if (!box || !btn) return;

            navigator.clipboard.writeText(box.value || "");

            const span = btn.querySelector('span');
            const target = span || btn;
            target.innerText = "Copied";
            setTimeout(() => { target.innerText = "Copy Negative Prompt"; }, 1200);
        }
        """
    )
