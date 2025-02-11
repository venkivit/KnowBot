import streamlit as st
from vector_store import VectorStore
from llm_handler import LLMHandler
from document_processor import DocumentProcessor
import time
import pandas as pd

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = VectorStore()
if 'llm_handler' not in st.session_state:
    st.session_state.llm_handler = LLMHandler()
if 'current_page' not in st.session_state:
    st.session_state.current_page = 0

# Page configuration
#st.set_page_config(page_title="Knowbot(Knowledge bot)", page_icon="🤖", layout="wide",initial_sidebar_state="collapsed")
st.set_page_config(page_title="Knowbot (Knowledge bot)",
                    page_icon="generated-icon.png", layout="wide",initial_sidebar_state="collapsed")
                    #menu_items=None)

# Custom CSS
st.markdown("""
    <style>
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        padding: 1rem;
    }
    .chat-message {
        display: flex;
        flex-direction: column;
        ##max-width: 80%;
        padding: 1rem;
        border-radius: 1rem;
        margin-bottom: 0.5rem;
    }
    .user-message {
        align-self: flex-end;
        background-color: #e6f3ff;
        border: 1px solid #b3d9ff;
        border-radius: 1rem 1rem 0 1rem;
    }
    .assistant-message {
        align-self: flex-start;
        background-color: #f0f2f6;
        border: 1px solid #d9d9d9;
        border-radius: 1rem 1rem 1rem 0;
    }
    .message-content {
        margin: 0;
        padding: 0.5rem;
    }
    .message-header {
        font-size: 0.8rem;
        color: #666;
        margin-bottom: 0.5rem;
    }
    .document-preview {
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        margin: 5px 0;
        background-color: white;
    }
    .stats-container {
        padding: 10px;
        background-color: #f8f9fa;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .upload-section {
        padding: 1rem;
        border: 2px dashed #ccc;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""",
            unsafe_allow_html=True)

# Main chat interface
st.title("🤖 Knowbot")
st.markdown(
    "Ask me anything! I'll search through the knowledge base and provide relevant answers."
)

# Chat container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Main chat container for all messages except the latest
for message in st.session_state.chat_history[:-1]:
    message_class = "user-message" if message["role"] == "user" else "assistant-message"
    st.markdown(f"""
        <div class="chat-message {message_class}">
            <div class="message-header">
                {"You" if message["role"] == "user" else "Assistant"}
            </div>
            <div class="message-content">
                {message["content"]}
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Add space for the last message and input form
st.markdown("<div style='height: 120px;'></div>", unsafe_allow_html=True)

# Display the latest message just above the input form
if st.session_state.chat_history:
    latest_message = st.session_state.chat_history[-1]
    message_class = "user-message" if latest_message["role"] == "user" else "assistant-message"
    st.markdown(f"""
        <div class="chat-message {message_class}" style="margin-bottom: 60px;">
            <div class="message-header">
                {"You" if latest_message["role"] == "user" else "Assistant"}
            </div>
            <div class="message-content">
                {latest_message["content"]}
            </div>
        </div>
    """, unsafe_allow_html=True)

# Create a placeholder for the input form
input_placeholder = st.empty()

# Move the input form to the bottom using custom CSS
st.markdown("""
    <style>
    [data-testid="stForm"] {
        position: fixed;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 650px;
        max-width: 85%;
        background-color: white;
        padding: 0.5rem;
        z-index: 1000;
        border-top: 1px solid #ddd;
        margin-bottom: 10px;
    }
    [data-testid="stForm"] input {
        height: 40px !important;
        padding: 8px !important;
    }
    @media (max-width: 768px) {
        [data-testid="stForm"] {
            width: 100%;
            left: 0;
            transform: none;
        }
    }
    </style>
""",
            unsafe_allow_html=True)

# Input form
with input_placeholder.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your message:", key="user_input")
    submit_button = st.form_submit_button("Send")

if submit_button and user_input:
    # Display user message immediately
    st.markdown(f"""
        <div class="chat-container">
            <div class="chat-message user-message">
                <div class="message-header">You</div>
                <div class="message-content">{user_input}</div>
            </div>
        </div>
    """,
                unsafe_allow_html=True)

    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })

    try:
        context = st.session_state.vector_store.query_documents(user_input)
        response_placeholder = st.empty()
        message_placeholder = st.markdown("")
        full_response = ""

        for response_chunk in st.session_state.llm_handler.generate_streaming_response(
                user_input, context,
            [{
                "user": msg["content"],
                "assistant": st.session_state.chat_history[i + 1]["content"]
            } for i, msg in enumerate(st.session_state.chat_history[:-1:2])]):
            full_response += response_chunk
            message_placeholder.markdown(f"""
                <div class="chat-message assistant-message">
                    <div class="message-header">Assistant</div>
                    <div class="message-content">{full_response}</div>
                </div>
            """,
                                         unsafe_allow_html=True)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": full_response
        })

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Sidebar with enhanced knowledge base management
with st.sidebar:
    st.header("Knowledge Base Management")

    # Display collection stats
    stats = st.session_state.vector_store.get_collection_stats()
    st.markdown("### 📊 Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Documents", stats.get("total_documents", 0))
    with col2:
        st.metric("Collection Size", f"{stats.get('total_documents', 0)} docs")

    # Add new documents
    st.markdown("### 📄 Add Documents")

    # File upload section with improved UI
    st.markdown("""
        <div class="upload-section">
            <p>📎 Upload your documents:</p>
            <small>Supported formats: PDF, TXT</small>
        </div>
    """,
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a file", type=['txt', 'pdf'])

    if uploaded_file:
        try:
            # Get file content and type
            content = uploaded_file.getvalue()
            file_type = uploaded_file.type.split('/')[
                -1]  # Get file extension from MIME type

            # Process the file using DocumentProcessor
            documents = DocumentProcessor.process_file(content, file_type)

            if documents:
                if st.session_state.vector_store.add_documents(documents):
                    st.success(
                        f"✅ Successfully added {len(documents)} documents from {uploaded_file.name}!"
                    )
                else:
                    st.error("❌ Failed to add documents.")
            else:
                st.warning("⚠️ No valid content found in the file.")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

    # Document browser
    st.markdown("### 📚 Document Browser")
    DOCS_PER_PAGE = 5

    # Get documents for current page
    docs, metadata = st.session_state.vector_store.get_documents(
        limit=DOCS_PER_PAGE,
        offset=st.session_state.current_page * DOCS_PER_PAGE)

    # Display documents
    for doc, meta in zip(docs, metadata):
        with st.expander(f"Document {meta.get('source', 'Untitled')}"):
            # Handle timestamp formatting with proper error checking
            timestamp = meta.get('timestamp', '')
            try:
                formatted_time = pd.Timestamp(timestamp).strftime(
                    '%Y-%m-%d %H:%M') if timestamp else 'No date'
            except (ValueError, TypeError):
                formatted_time = 'No date'

            st.markdown(f"""
                <div class="document-preview">
                    <small>Added: {formatted_time}</small>
                    <hr>
                    {doc[:200]}{'...' if len(doc) > 200 else ''}
                </div>
            """,
                        unsafe_allow_html=True)

            # Create unique key using both source and content hash
            doc_source = meta.get('source', '')
            if doc_source:
                delete_key = f"delete_{doc_source}_{hash(doc)}"
                if st.button("🗑️ Delete", key=delete_key):
                    if st.session_state.vector_store.delete_document(
                            doc_source):
                        st.success("Document deleted successfully!")
                        time.sleep(
                            1)  # Give time for the success message to be shown
                        st.rerun()
                    else:
                        st.error("Failed to delete document.")

    # Pagination
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.session_state.current_page > 0:
            if st.button("◀️ Previous"):
                st.session_state.current_page -= 1
                st.rerun()

    with col2:
        st.markdown(f"Page {st.session_state.current_page + 1}")

    with col3:
        if len(docs) == DOCS_PER_PAGE:
            if st.button("Next ▶️"):
                st.session_state.current_page += 1
                st.rerun()
