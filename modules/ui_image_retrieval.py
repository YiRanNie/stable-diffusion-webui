import gradio as gr

from modules import image_retrieval


def _toggle_search_source(source):
    is_online = (source == "Online Search")
    return [
        gr.update(visible=is_online),
        gr.update(visible=is_online),
        gr.update(visible=not is_online),
    ]


def _search_wrapper(query_image, search_source, keyword, local_folder, max_num):
    mode = "local" if search_source == "Local Folder" else "online"
    return image_retrieval.search_similar_images(query_image, mode, keyword, local_folder, max_num)


def _update_preview(img):
    return img


def create_ui():
    with gr.Blocks(analytics_enabled=False) as interface:
        gr.Markdown(
            """
            ## Image Similarity Search (SigLIP)
            Upload an image and find similar images from online sources or local folders.
            """
        )
        
        with gr.Tabs():
            with gr.Tab("Search"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Input")
                        
                        query_image = gr.Image(
                            label="Upload Query Image",
                            type="pil",
                            height=300,
                            interactive=True,
                            show_label=True
                        )
                        
                        with gr.Row():
                            image_preview = gr.Image(
                                label="Image Preview (Confirm Upload)",
                                type="pil",
                                height=150,
                                interactive=False,
                                show_label=True,
                                visible=True
                            )
                        
                        search_source = gr.Radio(
                            label="Search Source",
                            choices=["Online Search", "Local Folder"],
                            value="Online Search",
                            info="Choose to search from internet or local folder"
                        )
                        
                        keyword_input = gr.Textbox(
                            label="Search Keyword",
                            placeholder="e.g., cats, cyberpunk city, sunset beach",
                            lines=1,
                            visible=True
                        )
                        
                        max_num_slider = gr.Slider(
                            label="Number of Images to Crawl",
                            minimum=10,
                            maximum=50,
                            value=20,
                            step=5,
                            visible=True
                        )
                        
                        local_folder_input = gr.Textbox(
                            label="Local Folder Path",
                            placeholder="e.g., E:/Photos or C:/Users/YourName/Pictures",
                            lines=1,
                            visible=False
                        )
                        
                        search_btn = gr.Button(
                            "Search",
                            variant="primary"
                        )
                        
                        status_text = gr.Textbox(
                            label="Status",
                            lines=8,
                            interactive=False
                        )
                        
                        gr.Markdown(
                            """
                            ---
                            ### Tips
                            1. Upload a clear query image
                            2. Check preview area to confirm upload success
                            3. **Online Search**: English keywords work better (e.g., cats, sunset)
                            4. **Local Search**: Enter full folder path (e.g., E:/Photos)
                            5. Similarity is computed using SigLIP image features
                            6. First run requires model download, please wait
                            """
                        )
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### Results (Sorted by Similarity)")
                        
                        gallery_output = gr.Gallery(
                            label="Similar Images",
                            columns=4
                        )
                
                gr.Markdown("### Examples")
                gr.Examples(
                    examples=[
                        [None, "Online Search", "golden retriever dog", "", 20],
                        [None, "Online Search", "modern architecture", "", 25],
                        [None, "Online Search", "sunset beach", "", 30],
                    ],
                    inputs=[query_image, search_source, keyword_input, local_folder_input, max_num_slider],
                    label="Click example to auto-fill (upload image yourself)"
                )
            
            with gr.Tab("History"):
                gr.Markdown("## View Query History and Results")
                
                with gr.Row():
                    with gr.Column():
                        refresh_history_btn = gr.Button("Refresh History", variant="secondary")
                        history_display = gr.Markdown("Click refresh button to load history...")
                
                gr.Markdown("---")
                gr.Markdown("### View Specific Query Results")
                
                with gr.Row():
                    query_id_input = gr.Textbox(
                        label="Enter Query ID",
                        placeholder="e.g., 1",
                        lines=1
                    )
                    view_results_btn = gr.Button("View Results", variant="primary")
                
                results_info = gr.Textbox(
                    label="Query Info",
                    lines=5,
                    interactive=False
                )
                
                history_gallery = gr.Gallery(
                    label="History Query Results",
                    columns=4
                )
            
            with gr.Tab("Statistics"):
                gr.Markdown("## System Usage Statistics")
                
                refresh_stats_btn = gr.Button("Refresh Statistics", variant="secondary")
                stats_display = gr.Markdown("Click refresh button to load statistics...")
            
            with gr.Tab("Cache Management"):
                gr.Markdown("## Cache Management")
                gr.Markdown("Manage temporary files and storage limits for the image retrieval system.")
                
                with gr.Row():
                    refresh_cache_btn = gr.Button("Refresh Cache Status", variant="secondary")
                
                cache_status_display = gr.Markdown("Click refresh button to view cache status...")
                
                gr.Markdown("---")
                gr.Markdown("### Cleanup Actions")
                
                with gr.Row():
                    with gr.Column():
                        clear_temp_btn = gr.Button("Clear Temp Directory", variant="primary")
                        clear_temp_result = gr.Textbox(
                            label="Clear Temp Result",
                            lines=2,
                            interactive=False
                        )
                    
                    with gr.Column():
                        clear_old_btn = gr.Button("Clear Old Temp Files (24h+)", variant="primary")
                        clear_old_result = gr.Textbox(
                            label="Clear Old Files Result",
                            lines=2,
                            interactive=False
                        )
                
                with gr.Row():
                    with gr.Column():
                        clear_queries_btn = gr.Button("Clear Old Query Images", variant="primary")
                        keep_recent_slider = gr.Slider(
                            label="Keep Recent Query Images",
                            minimum=10,
                            maximum=500,
                            value=100,
                            step=10
                        )
                        clear_queries_result = gr.Textbox(
                            label="Clear Queries Result",
                            lines=2,
                            interactive=False
                        )
                    
                    with gr.Column():
                        enforce_limit_btn = gr.Button("Enforce Cache Limits", variant="stop")
                        enforce_limit_result = gr.Textbox(
                            label="Enforce Limits Result",
                            lines=4,
                            interactive=False
                        )
                
                gr.Markdown("---")
                gr.Markdown(
                    """
                    ### Cache Settings Info
                    - **Max Cache Size**: 500 MB (default)
                    - **Max Temp Files**: 1000 files
                    - **Max Query Images**: 500 images
                    - **Auto Cleanup Threshold**: 90% usage
                    - **Temp File Max Age**: 24 hours
                    """
                )
        
        query_image.change(
            fn=_update_preview,
            inputs=[query_image],
            outputs=[image_preview]
        )
        
        search_source.change(
            fn=_toggle_search_source,
            inputs=[search_source],
            outputs=[keyword_input, max_num_slider, local_folder_input]
        )
        
        search_btn.click(
            fn=_search_wrapper,
            inputs=[query_image, search_source, keyword_input, local_folder_input, max_num_slider],
            outputs=[gallery_output, status_text]
        )
        
        refresh_history_btn.click(
            fn=image_retrieval.load_query_history,
            outputs=[history_display]
        )
        
        view_results_btn.click(
            fn=image_retrieval.view_query_results,
            inputs=[query_id_input],
            outputs=[history_gallery, results_info]
        )
        
        refresh_stats_btn.click(
            fn=image_retrieval.load_statistics,
            outputs=[stats_display]
        )
        
        refresh_cache_btn.click(
            fn=image_retrieval.get_cache_status,
            outputs=[cache_status_display]
        )
        
        clear_temp_btn.click(
            fn=image_retrieval.clear_temp_cache,
            outputs=[clear_temp_result]
        )
        
        clear_old_btn.click(
            fn=image_retrieval.clear_old_temp_files,
            outputs=[clear_old_result]
        )
        
        clear_queries_btn.click(
            fn=image_retrieval.clear_old_query_images,
            inputs=[keep_recent_slider],
            outputs=[clear_queries_result]
        )
        
        enforce_limit_btn.click(
            fn=image_retrieval.enforce_cache_limits,
            outputs=[enforce_limit_result]
        )
        
        gr.Markdown(
            """
            ---
            <div style="text-align: center; color: #666; font-size: 0.9em;">
            Powered by SigLIP (Google) | Built with Gradio | Real-time Web Crawling
            </div>
            """
        )
    
    return interface
