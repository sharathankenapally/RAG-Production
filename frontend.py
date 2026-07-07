# pyrefly: ignore [missing-import]
import os
import uuid
import streamlit as st
import requests

# Set page title and layout
st.set_page_config(page_title="Document Chat Copilot", page_icon="💬", layout="centered")

# Custom CSS for modern chat bubble style and borders
st.markdown("""
<style>
    .stChatInput>div {
        border-radius: 8px;
    }
    .badge-container {
        display: flex;
        justify-content: flex-start;
        gap: 10px;
        margin-bottom: 10px;
        margin-top: 5px;
    }
    .badge {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        border: 1px solid #c8e6c9;
    }
    .badge-miss {
        background-color: #e3f2fd;
        color: #1565c0;
        border: 1px solid #bbdefb;
    }
    .eval-card {
        background-color: #f9f9f9;
        border: 1px solid #efefef;
        border-radius: 6px;
        padding: 10px 15px;
        margin-top: 8px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State variables
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: Knowledge Base Indicator
st.sidebar.title("📚 Knowledge Base")

raw_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
files_indexed = []
if os.path.exists(raw_dir):
    files_indexed = [f for f in os.listdir(raw_dir) if f.endswith(".pdf")]

# Initialize selected filters
selected_filters = None

if files_indexed:
    st.sidebar.write("**Currently indexed files:**")
    for f in files_indexed:
        st.sidebar.markdown(f"📄 `{f}`")
    
    # Multiselect for document filters
    st.sidebar.subheader("🔍 Search Focus")
    selected_filters = st.sidebar.multiselect(
        "Search only in these documents:",
        options=files_indexed,
        default=None,
        placeholder="All Documents"
    )
else:
    st.sidebar.info("No documents indexed. Please upload a PDF below.")

# Sidebar: File Uploader
st.sidebar.divider()
st.sidebar.subheader("📤 Upload Document")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file:", type=["pdf"])

if uploaded_file is not None:
    if st.sidebar.button("Ingest Uploaded PDF", type="secondary"):
        with st.sidebar.spinner("Uploading and indexing..."):
            try:
                # Send the file binary to the FastAPI backend
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                res = requests.post("http://localhost:8001/upload-pdf", files=files)
                
                if res.status_code == 200:
                    st.sidebar.success(f"Ingested {uploaded_file.name} successfully!")
                    st.rerun()
                else:
                    st.sidebar.error(res.json().get("detail", "Failed to ingest."))
            except Exception as e:
                st.sidebar.error(f"Error connecting to backend: {e}")

# Sidebar: Session Controls
st.sidebar.divider()
st.sidebar.subheader("⚙️ Chat Settings")
if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.sidebar.success("Chat history cleared!")
    st.rerun()

# Main Panel
st.title("💬 Smart Document Copilot")
st.write("Ask questions about your uploaded documents. The copilot will remember your chat history for follow-up questions.")

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        
        # Render badges and sources under assistant message
        if msg["role"] == "assistant":
            # Badges
            cached = msg.get("cached", False)
            if cached:
                st.markdown('<div class="badge-container"><div class="badge">⚡ Cached (Redis Hit)</div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge-container"><div class="badge badge-miss">🔄 Fresh LLM Response</div></div>', unsafe_allow_html=True)
            
            # Local Evaluation Display
            evaluation = msg.get("evaluation")
            if evaluation:
                f_score = evaluation.get("faithfulness", 100)
                r_score = evaluation.get("relevance", 100)
                reason = evaluation.get("reason", "")
                
                with st.container():
                    st.markdown('<div class="eval-card">', unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"🔬 **Faithfulness:** `{f_score}%`")
                        st.progress(f_score / 100.0)
                    with c2:
                        st.markdown(f"🎯 **Relevance:** `{r_score}%`")
                        st.progress(r_score / 100.0)
                    if reason:
                        st.caption(f"**Auditor Note:** {reason}")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Contexts expander
            contexts = msg.get("contexts", [])
            if contexts:
                with st.expander("Show retrieved document sources"):
                    for idx, ctx in enumerate(contexts):
                        st.markdown(f"**Source Passage #{idx + 1}:**\n{ctx}")

# Accept new chat input
if prompt := st.chat_input("Ask a question..."):
    # Render user prompt immediately
    with st.chat_message("user"):
        st.write(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Request answer from backend
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    "http://localhost:8001/query",
                    json={
                        "query": prompt,
                        "top_k": 3,
                        "session_id": st.session_state.session_id,
                        "filters": selected_filters if selected_filters else None
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    contexts = data["contexts"]
                    cached = data["cached"]
                    evaluation = data.get("evaluation")
                    
                    # Display Answer
                    message_placeholder.write(answer)
                    
                    # Add to session messages with extra metadata (contexts, cache status, evaluation)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "contexts": contexts,
                        "cached": cached,
                        "evaluation": evaluation
                    })
                    st.rerun()
                else:
                    detail = response.json().get("detail", "Error connecting to service.")
                    message_placeholder.error(f"Error {response.status_code}: {detail}")
            except Exception as e:
                message_placeholder.error(f"Failed to connect to backend: {e}")
