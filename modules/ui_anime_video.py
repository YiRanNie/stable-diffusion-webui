import gradio as gr

from modules import anime_video


def _variant_label_map():
    return {
        "face_paint_512_v2": "face_paint_512_v2 (portrait, soft skin)",
        "face_paint_512_v1": "face_paint_512_v1 (portrait, classic)",
        "paprika": "paprika (anime line-art)",
    }


def _variant_choices():
    mapping = _variant_label_map()
    return [f"{k} – {v.split('(', 1)[1][:-1]}" for k, v in mapping.items()]


def _label_to_variant(label: str) -> str:
    mapping = _variant_label_map()
    for k, v in mapping.items():
        if label.startswith(k):
            return k
    return anime_video.DEFAULT_VARIANT


def _build_variant_dropdown_choices():
    mapping = _variant_label_map()
    return [f"{k} – {mapping[k].split('(', 1)[1][:-1]}" for k in anime_video.VALID_VARIANTS]


def _variant_display_to_key(display: str) -> str:
    if not isinstance(display, str):
        return anime_video.DEFAULT_VARIANT
    key = display.split("–", 1)[0].strip()
    if key in anime_video.VALID_VARIANTS:
        return key
    return anime_video.DEFAULT_VARIANT


def _init_download_state():
    status = anime_video.get_download_status()
    return status


def _on_style_change(display_value: str):
    variant = _variant_display_to_key(display_value)
    status = _init_download_state()
    is_ready = status.get(variant, False)

    if is_ready:
        button_update = gr.Button.update(visible=False)
        text = f"<span style='color: #16a34a;'>Model '{variant}' is ready to use.</span>"
    else:
        button_update = gr.Button.update(visible=True)
        text = f"<span style='color: #dc2626;'>Model '{variant}' is not downloaded.</span>"

    return button_update, text


def _on_download_click(display_value: str, progress=gr.Progress(track_tqdm=False)):
    variant = _variant_display_to_key(display_value)
    message = anime_video.gradio_download_variant(variant, progress)

    status = _init_download_state()
    is_ready = status.get(variant, False)

    if is_ready:
        button_update = gr.Button.update(visible=False)
        text = f"<span style='color: #16a34a;'>{message}</span>"
    else:
        button_update = gr.Button.update(visible=True)
        text = f"<span style='color: #dc2626;'>{message}</span>"

    return button_update, text


def _run_video2anime(video, max_side, style_display, strength, smoothing):
    variant = _variant_display_to_key(style_display)
    return anime_video.gradio_animegan(video, max_side, variant, strength, smoothing)


def create_ui():
    style_choices = _build_variant_dropdown_choices()
    default_display = style_choices[0]

    with gr.Blocks(analytics_enabled=False) as interface:
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                video_in = gr.Video(
                    label="Input video",
                    elem_id="video2anime_input",
                )

                max_side = gr.Slider(
                    minimum=360,
                    maximum=1280,
                    value=720,
                    step=40,
                    label="Max resolution (longer side, pixels)",
                    elem_id="video2anime_max_side",
                )

                style_variant = gr.Dropdown(
                    choices=style_choices,
                    value=default_display,
                    label="Style variant",
                    elem_id="video2anime_style",
                )

                style_strength = gr.Slider(
                    minimum=0.0,
                    maximum=1.5,
                    value=1.0,
                    step=0.05,
                    label="style strength",
                    elem_id="video2anime_strength",
                )

                temporal_smoothing = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.05,
                    label="temporal smoothing factor",  
                    elem_id="video2anime_temporal_smoothing_v3", 
                )


                with gr.Row():
                    download_button = gr.Button(
                        "Download model",
                        elem_id="video2anime_download_button",
                    )
                    download_status = gr.HTML(
                        "",
                        elem_id="video2anime_download_status",
                    )

                convert_button = gr.Button(
                    "Convert to anime video",
                    elem_id="video2anime_run",
                )

            with gr.Column(scale=1):
                video_out = gr.Video(
                    label="Anime-style output",
                    elem_id="video2anime_output",
                )

        initial_status = _init_download_state()
        if initial_status.get(anime_video.DEFAULT_VARIANT, False):
            download_button.visible = False

        style_variant.change(
            fn=_on_style_change,
            inputs=[style_variant],
            outputs=[download_button, download_status],
        )

        download_button.click(
            fn=_on_download_click,
            inputs=[style_variant],
            outputs=[download_button, download_status],
        )

        convert_button.click(
            fn=_run_video2anime,
            inputs=[video_in, max_side, style_variant, style_strength, temporal_smoothing],
            outputs=[video_out],
        )

    return interface
