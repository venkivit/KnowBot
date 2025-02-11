import streamlit as st
from vector_store import VectorStore
from llm_handler import LLMHandler
import time

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = VectorStore()
if 'llm_handler' not in st.session_state:
    st.session_state.llm_handler = LLMHandler()

# Page configuration
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stTextInput {
        padding: 10px;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e6f3ff;
        border: 1px solid #b3d9ff;
    }
    .assistant-message {
        background-color: #f0f2f6;
        border: 1px solid #d9d9d9;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🤖 AI Chatbot")
st.markdown("Ask me anything! I'll search through the knowledge base and provide relevant answers.")

# Chat interface
for message in st.session_state.chat_history:
    message_class = "user-message" if message["role"] == "user" else "assistant-message"
    st.markdown(f"""
        <div class="chat-message {message_class}">
            <b>{"You" if message["role"] == "user" else "Assistant"}:</b>
            {message["content"]}
        </div>
    """, unsafe_allow_html=True)

# Input form
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your message:", key="user_input")
    submit_button = st.form_submit_button("Send")

if submit_button and user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })
    
    # Show loading spinner
    with st.spinner("Thinking..."):
        try:
            # Query vector store for relevant context
            context = st.session_state.vector_store.query_documents(user_input)
            
            # Generate response using LLM
            response = st.session_state.llm_handler.generate_response(
                user_input,
                context,
                [{"user": msg["content"], "assistant": st.session_state.chat_history[i+1]["content"]}
                 for i, msg in enumerate(st.session_state.chat_history[:-1:2])]
            )
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response
            })
            
            # Rerun to update chat display
            st.rerun()
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Sidebar with document management
with st.sidebar:
    st.header("Knowledge Base Management")
    
    # Document count
    doc_count = st.session_state.vector_store.get_document_count()
    st.metric("Documents in Knowledge Base", doc_count)
    
    # Add new documents
    st.subheader("Add New Documents")
    uploaded_file = st.file_uploader("Upload text file", type=['txt'])
    
    if uploaded_file:
        try:
            content = uploaded_file.getvalue().decode()
            documents = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            if st.session_state.vector_store.add_documents(documents):
                st.success(f"Successfully added {len(documents)} documents!")
            else:
                st.error("Failed to add documents.")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
