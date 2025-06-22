import gradio as gr
import sys
import io
import os
import time
import logging
from rag_cmd import LateFusionHybridRAGChatPipeline
import utils
from ansi2html import Ansi2HTMLConverter

# --- Globals & Setup ---
conv = Ansi2HTMLConverter(dark_bg=True, scheme="xterm")
# Extract the CSS for ANSI colors from the converter's generated <style> tag
style_tag = conv.produce_headers()
ansi_css = style_tag.replace('<style type="text/css">', '').replace('</style>', '').strip()


# --- Custom Logger to Tee stdout ---
class Tee:
    """Helper class to redirect stdout to both console and a string stream."""
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# --- Backend Functions ---

def setup_pipeline(
    source_path, chunk_dst, dense_index, sparse_index,
    proc_batch, upsert_batch, sparse_top_k,
    dense_k, sparse_k, rerank_k,
    fusion, alpha, rrf_k, new_embeds,use_reranker, run_initial_setup,
    progress=gr.Progress(track_tqdm=True)
):
    """
    Initializes the RAG pipeline and converts logs to styled HTML.
    """
    log_stream = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_stream)

    # Capture logs from the pipeline's logger
    pipeline_logger = logging.getLogger("rag3_late_fuse_graph_chat_multi_followUP")
    stream_handler = logging.StreamHandler(log_stream)
    original_level = pipeline_logger.level
    pipeline_logger.setLevel(logging.INFO) # Ensure INFO logs are captured
    pipeline_logger.addHandler(stream_handler)

    try:
        progress(0, desc="Building Configuration...")
        config = {
            'source_path': source_path, 'chunk_dst': chunk_dst,
            'dense_index_name': dense_index, 'sparse_index_name': sparse_index,
            'processing_batch_size': int(proc_batch), 'upsert_batch_size': int(upsert_batch),
            'sparse_top_k': int(sparse_top_k), 'dense_search_top_k': int(dense_k),
            'sparse_search_top_k': int(sparse_k), 'rerank_top_k': int(rerank_k),
            'fusion_method': fusion, 'alpha': alpha, 'rrf_k': int(rrf_k),
            'create_new_embeddings': new_embeds, 'max_chat_history': 20,
            'max_iterations': 15, 'max_time_seconds': 60.0,
            'use_reranker': use_reranker,
            'run_initial_setup': run_initial_setup,
        }
        config['reset_db'] = True if config['create_new_embeddings'] else False

        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        progress(0.1, desc="Initializing Pipeline...")
        pipeline = LateFusionHybridRAGChatPipeline(config)
        progress(0.2, desc="Loading Models...")
        pipeline.load_models()
        
        if config['run_initial_setup']:
            progress(0.4, desc="Creating/Loading Embeddings...")
            if config['create_new_embeddings']:
                utils.checkPath(config['chunk_dst'])
                dense_embeddings, sparse_embeddings = pipeline.create_embeddings(config['source_path'], config['chunk_dst'])
            else:
                dense_embeddings, sparse_embeddings = pipeline.load_embeddings(config['chunk_dst'])
            if not dense_embeddings or not sparse_embeddings:
                raise ValueError("No embeddings found. Check paths or enable 'Create New Embeddings'.")
            progress(0.6, desc="Initializing Indices...")
            pipeline.initialize_indices(config['dense_index_name'], config['sparse_index_name'], 1024, config['sparse_top_k'], config['reset_db'])
            progress(0.7, desc="Populating Indices...")
            pipeline.populate_indices(dense_embeddings, sparse_embeddings)
        else:
            # If not running setup, we still need to connect to the indices without resetting them.
            progress(0.6, desc="Connecting to Existing Indices...")
            pipeline.initialize_indices(config['dense_index_name'], config['sparse_index_name'], 1024, config['sparse_top_k'], reset_db=False)

        progress(0.9, desc="Building Graph...")
        pipeline.build_graph()
        graph_path = "images/rag_system_graph.png"
        pipeline.save_graph_visualization(graph_path)
        status = "✅ Setup Complete! You can now go to the 'Chat' tab."
        print(status)
        full_log = log_stream.getvalue()
        html_log = conv.convert(full_log, full=False)
        return pipeline, f"<pre>{html_log}</pre>", status, gr.update(value=graph_path, visible=True), gr.update(visible=False)
    except Exception as e:
        print(f"\n--- GRADIO APP ERROR ---\nAn error occurred during pipeline setup: {e}\n--------------------------")
        error_log = log_stream.getvalue()
        html_error_log = conv.convert(error_log, full=False)
        return None, f"<pre>{html_error_log}</pre>", "❌ Setup Failed. Check Logs tab for details.", gr.update(visible=False), gr.update(visible=True)
    finally:
        # Clean up the logger and stdout
        pipeline_logger.removeHandler(stream_handler)
        pipeline_logger.setLevel(original_level)
        sys.stdout = original_stdout

def chat_responder(pipeline_state, message, chat_history, log_history, setup_status_text):
    """
    Handles a chat interaction, converts new logs, and handles pipeline reset on 'goodbye'.
    """
    if not pipeline_state:
        chat_history.append((message, "Pipeline not initialized. Please go to the 'Configuration' tab and click 'Initialize Pipeline'."))
        return chat_history, log_history, pipeline_state, "Not initialized."

    log_stream = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_stream)

    new_pipeline_state = pipeline_state
    new_status = setup_status_text

    try:
        result = pipeline_state.run_chat_query(message, fusion_method=pipeline_state.config['fusion_method'], alpha=pipeline_state.config['alpha'], rrf_k=pipeline_state.config['rrf_k'])
        bot_message = result['generation']

        # If 'goodbye' was detected, de-initialize the pipeline
        if result.get('session_end', False):
            print("\n--- GRADIO APP INFO ---\n'goodbye' detected. De-initializing pipeline state.")
            pipeline_state.deinitialize()
            new_pipeline_state = None
            new_status = "Pipeline reset. Re-initialize to start a new session."
            bot_message += f"\n\n**System Reset:** {new_status}"

    except Exception as e:
        bot_message = f"An error occurred: {e}"
        print(f"\n--- GRADIO APP ERROR ---\n{bot_message}")
    finally:
        sys.stdout = original_stdout

    query_logs = log_stream.getvalue()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    header = f"--- Log at {timestamp} for query: '{message}' ---\n"
    full_entry_raw = header + query_logs
    new_log_html = conv.convert(full_entry_raw, full=False)
    
    updated_logs = f"<div><pre>{new_log_html}</pre></div><hr/>" + (log_history or "")
    chat_history.append((message, bot_message))
    
    return chat_history, updated_logs, new_pipeline_state, new_status

# --- Gradio UI ---

# Combine our custom CSS with the CSS from ansi2html
css = f"""
{ansi_css}

#log-output {{
    background-color: #1E1E1E;
    color: #D4D4D4;
    font-family: 'Consolas', 'Courier New', monospace;
    padding: 1rem;
    border-radius: 8px;
    font-size: 0.9em;
    overflow-y: auto;
    height: 600px;
}}
#log-output pre {{
    white-space: pre-wrap;
    word-wrap: break-word;
    background-color: transparent !important;
    color: inherit !important;
    padding: 0 !important;
    margin: 0 !important;
}}
#log-output hr {{
    border-color: #444;
    margin: 1rem 0;
}}
"""

with gr.Blocks(theme=gr.themes.Soft(), title="AgenticRAG Chat", css=css) as demo:
    pipeline_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=3, min_width=100):
           gr.Image("images/logo.png", height=30, show_label=False, show_download_button=False, show_fullscreen_button=False, interactive=False, container=False)
        with gr.Column(scale=8):
            gr.Markdown("# AgenticRAG with Late Fusion and Multi-Document Support")
    
    with gr.Tabs():
        with gr.TabItem("Chat", id=0):
            chatbot = gr.Chatbot(
                [], elem_id="chatbot", bubble_full_width=False, height=600,
                avatar_images=("images/user.png", "images/bot.png"),
            )
            with gr.Row():
                chat_input = gr.Textbox(show_label=False, placeholder="Enter your query here...", lines=2, scale=4)
                send_button = gr.Button("Send", variant="primary", scale=1)
            gr.Examples(
                examples=[
                    "There is any deadline for the submission?",
                    "who can sign the lear ?",
                    "How many types of deadlines are there in the call ?",
                    "What can you do for me ?",
                    "bye bye 👋"
                ],
                inputs=chat_input,
                label="Example Queries"
            )
                
        with gr.TabItem("Logs", id=1):
            with gr.Accordion("Execution Logs", open=True):
                log_output = gr.HTML(elem_id="log-output", value="<pre>Logs will appear here after initialization.</pre>")
            with gr.Accordion("System Graph", open=False):
                graph_display = gr.Image(label="RAG System Flow Graph", visible=False, show_label=True)
                graph_placeholder = gr.Markdown("*Graph will be displayed after successful initialization.*", visible=True)

        with gr.TabItem("Configuration", id=2):
            gr.Markdown("## Pipeline Configuration")
            gr.Markdown("Adjust settings and initialize the system. You must do this before chatting.")
            
            setup_status = gr.Textbox(label="Setup Status", interactive=False, value="Not initialized.")
            initialize_button = gr.Button("🚀 Connect!", variant="primary")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Paths & Data")
                    source_path_input = gr.Textbox(label="Source Path (Folder with documents)", value="../preprocessing/cropsAll/")
                    chunk_dst_input = gr.Textbox(label="Chunk Destination Path", value="chunks_grants/multi_doc/")
                    new_embeds_input = gr.Checkbox(label="Create New Embeddings", value=False)
                    run_initial_setup_input = gr.Checkbox(label="Run Initial Setup (Embeddings & Indexing)", value=False)
                    gr.Markdown("### Index Names")
                    dense_index_input = gr.Textbox(label="Dense Index Name", value="grant-dense-multi-doc")
                    sparse_index_input = gr.Textbox(label="Sparse Index Name", value="grant-sparse-multi-doc")
                with gr.Column():
                    gr.Markdown("### Processing & Search")
                    proc_batch_input = gr.Slider(label="Processing Batch Size", minimum=1, maximum=10, value=1, step=1)
                    upsert_batch_input = gr.Slider(label="Upsert Batch Size", minimum=10, maximum=200, value=10, step=10)
                    sparse_top_k_input = gr.Slider(label="Sparse Top-K (Embedding)", minimum=100, maximum=2000, value=1000, step=50)
                    dense_k_input = gr.Slider(label="Dense Search Top-K", minimum=1, maximum=20, value=5, step=1)
                    sparse_k_input = gr.Slider(label="Sparse Search Top-K", minimum=1, maximum=20, value=5, step=1)
                    rerank_k_input = gr.Slider(label="Rerank Top-K", minimum=1, maximum=10, value=3, step=1)
                    use_reranker_input = gr.Checkbox(label="Use Reranker", value=False)
                with gr.Column():
                    gr.Markdown("### Fusion Method")
                    fusion_input = gr.Dropdown(label="Fusion Method", choices=["rrf", "score_based"], value="rrf")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Fusion Method")
                    fusion_input = gr.Dropdown(label="Fusion Method", choices=["rrf", "score_based"], value="rrf")
                    alpha_input = gr.Slider(label="Alpha (for score_based fusion)", minimum=0.0, maximum=1.0, value=0.7, step=0.05)
                    rrf_k_input = gr.Slider(label="RRF k (for rrf fusion)", minimum=10, maximum=100, value=60, step=5)

    # --- Event Handlers ---
    config_inputs = [
        source_path_input, chunk_dst_input, dense_index_input, sparse_index_input,
        proc_batch_input, upsert_batch_input, sparse_top_k_input,
        dense_k_input, sparse_k_input, rerank_k_input,
        fusion_input, alpha_input, rrf_k_input, new_embeds_input, use_reranker_input,
        run_initial_setup_input
    ]
    config_outputs = [pipeline_state, log_output, setup_status, graph_display, graph_placeholder]
    initialize_button.click(fn=setup_pipeline, inputs=config_inputs, outputs=config_outputs)

    chat_submit_args = {
        "fn": chat_responder, 
        "inputs": [pipeline_state, chat_input, chatbot, log_output, setup_status], 
        "outputs": [chatbot, log_output, pipeline_state, setup_status]
    }
    chat_input.submit(**chat_submit_args).then(lambda: gr.update(value=""), None, [chat_input], queue=False)
    send_button.click(**chat_submit_args).then(lambda: gr.update(value=""), None, [chat_input], queue=False)

if __name__ == "__main__":
    demo.queue().launch(debug=True, share=True, favicon_path="images/favicon.ico")