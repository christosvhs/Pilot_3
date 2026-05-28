import os
import time
import gradio as gr
import requests

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")
API_URL = f"{API_BASE}/query"
SWITCH_URL = f"{API_BASE}/switch_llm"
HEALTH_URL = f"{API_BASE}/health"

_LLM_LABELS = {
    "salamandra": "Salamandra-7B-Instruct",
    "krikri": "Llama-Krikri-8B-Instruct",
}

_WELCOME_MD = """
## Welcome to the Call Center Assistant!

I can help you with:
- **General Inquiries**: Answer questions about products, services, and policies
- **Account Support**: Help with account-related questions and information
- **Technical Assistance**: Guide you through technical or procedural issues
- **Information Lookup**: Retrieve relevant information from the knowledge base
"""

_ARCHITECTURE_MD = """
**Architecture:**
- Non-transactional: RAG + LLM
- Transactional: NER + LLM + API
- Response Analysis: XLM-T / GLiNER2
"""

_CSS = """
#send-btn { background-color: #5B5EA6; color: white; }
#header { text-align: center; }
#docs-panel textarea { font-family: monospace; font-size: 12px; }
.spinner {
    display: inline-block;
    width: 14px;
    height: 14px;
    border: 3px solid #e0e0e0;
    border-top: 3px solid #5B5EA6;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    vertical-align: middle;
    margin-right: 8px;
}
@keyframes spin {
    0%   { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
"""


def _call_pipeline(dialog_history: list[str], retriever_type: str, llm_type: str) -> dict:
    try:
        r = requests.post(
            API_URL,
            json={
                "dialog_history": dialog_history,
                "retriever_type": retriever_type,
                "llm_type": llm_type,
            },
            timeout=300,  # longer to accommodate LLM swap
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"response": "Error: could not reach the pipeline server. Is server.py running?",
                "retrieved_documents": {}}
    except Exception as e:
        return {"response": f"Error: {e}", "retrieved_documents": {}}


def _format_docs(retrieved: dict) -> str:
    docs = retrieved.get("documents", [[]])
    dists = retrieved.get("distances", [[]])
    if not docs or not docs[0]:
        return ""
    docs, dists = docs[0], dists[0]
    return "\n\n".join(
        f"[{i + 1}] distance={d:.4f}\n{doc}" for i, (doc, d) in enumerate(zip(docs, dists))
    )


def send_message(
    message: str,
    chatbot: list,
    dialog_history: list[str],
    retriever_type: str,
    llm_type: str,
):
    message = message.strip()
    if not message:
        return chatbot, dialog_history, "", ""

    dialog_history = dialog_history + [message]
    result = _call_pipeline(dialog_history, retriever_type, llm_type)
    response = result["response"]
    docs_text = _format_docs(result.get("retrieved_documents", {}))
    dialog_history = dialog_history + [response]

    chatbot = chatbot + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response},
    ]
    return chatbot, dialog_history, "", docs_text


def clear_chat():
    return [], [], "", ""


def _ui_state(status: str, interactive: bool):
    return (
        status,
        gr.update(interactive=interactive),  # msg_input
        gr.update(interactive=interactive),  # send_btn
        gr.update(interactive=interactive),  # clear_btn
        gr.update(interactive=interactive),  # llm_choice
        gr.update(interactive=interactive),  # retriever_choice
    )


def wait_for_server():
    yield _ui_state(
        '<span class="spinner"></span>Starting server (loading initial LLM)...',
        interactive=False,
    )

    deadline = time.time() + 300  # 5 minutes
    while time.time() < deadline:
        try:
            r = requests.get(HEALTH_URL, timeout=2)
            if r.status_code == 200:
                llm_key = r.json().get("current_llm", "")
                label = _LLM_LABELS.get(llm_key, llm_key)
                yield _ui_state(f"✅ {label} ready", interactive=True)
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)

    yield _ui_state(
        "❌ Server did not become ready within 5 minutes",
        interactive=False,
    )


def switch_llm(llm_type: str):
    label = _LLM_LABELS.get(llm_type, llm_type)
    loading = (
        f'<span class="spinner"></span>'
        f'Loading {label}... (this can take 1–2 minutes)'
    )
    yield _ui_state(loading, interactive=False)
    try:
        r = requests.post(SWITCH_URL, json={"llm_type": llm_type}, timeout=300)
        r.raise_for_status()
        yield _ui_state(f"✅ {label} ready", interactive=True)
    except Exception as e:
        yield _ui_state(f"❌ Failed to load {label}: {e}", interactive=True)


with gr.Blocks(theme=gr.themes.Soft(), css=_CSS, title="Call Center Assistant") as demo:

    dialog_history = gr.State([])

    gr.Markdown(
        "# 🏢 Call Center Assistant — Powered by LLM + RAG + Advanced NLP",
        elem_id="header",
    )
    gr.Markdown(_WELCOME_MD)

    with gr.Row():

        # ── Left: chat ──────────────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Assistant")
            chatbot = gr.Chatbot(height=500, show_label=False)

            with gr.Row():
                msg_input = gr.Textbox(
                    label="Your Message",
                    placeholder="Type your message here...",
                    lines=2,
                    scale=4,
                    show_label=True,
                )
                send_btn = gr.Button("Send 🚀", variant="primary", scale=1, elem_id="send-btn")

            clear_btn = gr.Button("Clear Chat 🗑️")

        # ── Right: monitoring ───────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 🔍 System Monitoring")

            retriever_choice = gr.Radio(
                choices=[
                    ("Fine-tuned LaBSE", "finetuned"),
                    ("Baseline LaBSE (out-of-the-box)", "baseline"),
                ],
                value="finetuned",
                label="Retriever",
            )

            llm_choice = gr.Radio(
                choices=[
                    ("Salamandra-7B-Instruct", "salamandra"),
                    ("Llama-Krikri-8B-Instruct", "krikri"),
                ],
                value="krikri",
                label="LLM (switching reloads the model — first query takes ~1–2 min)",
            )
            llm_status = gr.HTML("✅ Llama-Krikri-8B-Instruct ready")

            docs_panel = gr.Textbox(
                label="Retrieved Documents",
                lines=20,
                max_lines=30,
                interactive=False,
                elem_id="docs-panel",
                placeholder="Retrieved documents will appear here after sending a message.",
            )

            gr.Markdown(_ARCHITECTURE_MD)

    send_btn.click(
        send_message,
        inputs=[msg_input, chatbot, dialog_history, retriever_choice, llm_choice],
        outputs=[chatbot, dialog_history, msg_input, docs_panel],
    )
    msg_input.submit(
        send_message,
        inputs=[msg_input, chatbot, dialog_history, retriever_choice, llm_choice],
        outputs=[chatbot, dialog_history, msg_input, docs_panel],
    )
    clear_btn.click(
        clear_chat,
        outputs=[chatbot, dialog_history, msg_input, docs_panel],
    )

    llm_choice.change(
        switch_llm,
        inputs=[llm_choice],
        outputs=[llm_status, msg_input, send_btn, clear_btn, llm_choice, retriever_choice],
    )

    demo.load(
        wait_for_server,
        outputs=[llm_status, msg_input, send_btn, clear_btn, llm_choice, retriever_choice],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=19000)
