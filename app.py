import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import re
import os

# CSS styles
css = '''
<style>
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.chat-message.user {
    background-color: #2b313e
}

.chat-message.bot {
    background-color: #475063
}

.chat-message .avatar {
    width: 15%;
    display: flex;
    align-items: center;
    justify-content: center;
}

.chat-message .avatar i {
    font-size: 2.5rem;
    color: #fff;
}

.chat-message .message {
    width: 85%;
    padding: 0 1.5rem;
    color: #fff;
    line-height: 1.6;
}

.stButton>button {
    width: 100%;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <i class="fas fa-robot"></i>
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <i class="fas fa-user"></i>
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            st.warning(f"Error reading {pdf.name}: {str(e)}")
    return text

def get_text_chunks(text):
    """Split text into chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    """Create FAISS vector store from text chunks."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    """Create the RAG conversation chain."""
    
    # Try to get HuggingFace token (works for both local and Streamlit Cloud)
    try:
        # Try Streamlit secrets first (for Streamlit Cloud)
        hf_token = st.secrets.get("HF_TOKEN")
    except:
        # Fall back to environment variables (for local development)
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    if not hf_token:
        st.error("⚠️ HuggingFace API token not found!")
        st.info("Please set HF_TOKEN in your .env file or environment variables.")
        st.code("HF_TOKEN=your_token_here", language="bash")
        st.markdown("Get your token from: https://huggingface.co/settings/tokens")
        return None
    
    try:
        from huggingface_hub import InferenceClient
        
        # Create inference client
        client = InferenceClient(token=hf_token)
        
        # Create a custom LLM wrapper
        class HuggingFaceInferenceLLM:
            def __init__(self, client, model_id="google/flan-t5-large"):
                self.client = client
                self.model_id = model_id
            
            def invoke(self, prompt):
                try:
                    response = self.client.text_generation(
                        prompt,
                        model=self.model_id,
                        max_new_tokens=512,
                        temperature=0.5,
                        return_full_text=False
                    )
                    return response
                except Exception as e:
                    return f"Error: {str(e)}"
        
        llm = HuggingFaceInferenceLLM(client)
        
        # Enhanced prompt template
        template = """You are a helpful assistant that answers questions based on the provided context from PDF documents.

Context from documents:
{context}

Question: {question}

Instructions:
- Answer the question based ONLY on the context provided above
- If the answer is not in the context, say "I don't have enough information in the documents to answer this question."
- Be concise but complete
- Use specific details from the context when possible

Answer:"""
        
        prompt = PromptTemplate.from_template(template)
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
        )
        
        def format_docs(docs):
            """Format retrieved documents."""
            formatted = []
            for i, doc in enumerate(docs, 1):
                formatted.append(f"[Excerpt {i}]\n{doc.page_content}")
            return "\n\n".join(formatted)
        
        # Create chain manually
        def run_chain(question):
            try:
                # Get relevant documents
                docs = retriever.invoke(question)  # Changed from get_relevant_documents
                
                if not docs:
                    return "I couldn't find any relevant information in the documents to answer your question."
                
                context = format_docs(docs)
                
                # Format prompt
                full_prompt = template.format(context=context, question=question)
                
                # Get response
                response = llm.invoke(full_prompt)
                
                return response
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                st.error(f"Error in chain execution: {str(e)}")
                st.code(error_details, language="python")
                return f"Sorry, I encountered an error: {str(e)}"
        
        return run_chain
        
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        return None

def is_greeting(text):
    """Check if the input is a greeting."""
    greetings = r"\b(hi|hello|hey|greetings|good morning|good afternoon|good evening)\b"
    return bool(re.search(greetings, text.lower().strip()))

def handle_greeting(greeting):
    """Generate appropriate greeting response."""
    greeting_lower = greeting.lower().strip()
    responses = {
        "hi": "Hi there! 👋 How can I help you with your documents today?",
        "hello": "Hello! 👋 I'm ready to assist you with any questions about your PDFs.",
        "hey": "Hey! 👋 What would you like to know about your documents?",
        "greetings": "Greetings! 👋 I'm here to help you with your PDF queries.",
        "good morning": "Good morning! ☀️ How may I assist you with your documents today?",
        "good afternoon": "Good afternoon! 🌤️ What questions do you have about your PDFs?",
        "good evening": "Good evening! 🌙 I'm here to help with any document-related questions."
    }
    
    for key in responses:
        if key in greeting_lower:
            return responses[key]
    
    return "Hello! 👋 How can I assist you with your documents today?"

def handle_userinput(user_question):
    """Process user input and generate response."""
    
    # Handle greetings
    if is_greeting(user_question):
        response = handle_greeting(user_question)
        st.session_state.chat_history.append(("user", user_question))
        st.session_state.chat_history.append(("bot", response))
        display_chat_history()
        return

    # Check if documents are processed
    if "conversation" not in st.session_state or st.session_state.conversation is None:
        st.error("⚠️ Please upload and process your PDF documents first before asking questions.")
        return

    try:
        # Show loading indicator
        with st.spinner("🔍 Searching through your documents..."):
            # Get response from chain (now it's a function, not a chain object)
            answer = st.session_state.conversation(user_question)
            
            # Clean up the answer
            answer = answer.strip()
            
            # Update chat history
            st.session_state.chat_history.append(("user", user_question))
            st.session_state.chat_history.append(("bot", answer))
        
        # Display all messages
        display_chat_history()
        
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        st.error(error_msg)
        
        # Provide helpful troubleshooting tips
        with st.expander("🔧 Troubleshooting Tips"):
            st.markdown("""
            **Common issues:**
            1. **API Token**: Make sure your HuggingFace token is valid
            2. **Rate Limits**: Free tier has usage limits - wait a moment and try again
            3. **Model Loading**: The model might be loading for the first time (can take 1-2 minutes)
            4. **Network**: Check your internet connection
            
            **To fix:**
            - Verify your HF_TOKEN in the .env file
            - Try asking a simpler question
            - Wait a few seconds and try again
            - Check HuggingFace status: https://status.huggingface.co/
            """)

def display_chat_history():
    """Display all chat messages."""
    for role, message in st.session_state.chat_history:
        if role == "user":
            st.write(user_template.replace("{{MSG}}", message), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message), unsafe_allow_html=True)

def clear_chat():
    """Clear chat history and conversation."""
    st.session_state.chat_history = []
    st.session_state.conversation = None
    st.session_state.processed_docs = False

def main():
    """Main application function."""
    load_dotenv()
    
    st.set_page_config(
        page_title="Chat with PDFs",
        page_icon="📚",
        layout="wide"
    )
    st.write(css, unsafe_allow_html=True)

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_docs" not in st.session_state:
        st.session_state.processed_docs = False

    # Header
    st.title("📚 Chat with Multiple PDFs")
    st.markdown("Upload your PDF documents and ask questions about their content!")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Document Management")
        
        # File uploader
        pdf_docs = st.file_uploader(
            "Upload your PDFs here",
            accept_multiple_files=True,
            type=['pdf'],
            help="You can upload multiple PDF files"
        )
        
        # Show uploaded files
        if pdf_docs:
            st.success(f"✅ {len(pdf_docs)} file(s) uploaded")
            with st.expander("📄 Uploaded Files"):
                for pdf in pdf_docs:
                    st.text(f"• {pdf.name}")
        
        # Process button
        if st.button("🚀 Process Documents", type="primary"):
            if not pdf_docs:
                st.error("❌ Please upload at least one PDF document.")
            else:
                with st.spinner("Processing your documents..."):
                    try:
                        # Extract text
                        st.info("📖 Extracting text from PDFs...")
                        raw_text = get_pdf_text(pdf_docs)
                        
                        if not raw_text.strip():
                            st.error("❌ No text could be extracted from the PDFs. They might be image-based or empty.")
                            return
                        
                        # Create chunks
                        st.info("✂️ Splitting text into chunks...")
                        text_chunks = get_text_chunks(raw_text)
                        st.success(f"Created {len(text_chunks)} text chunks")
                        
                        # Create vector store
                        st.info("🧠 Creating vector database...")
                        vectorstore = get_vectorstore(text_chunks)
                        
                        # Create conversation chain
                        st.info("🔗 Setting up conversation chain...")
                        conversation_chain = get_conversation_chain(vectorstore)
                        
                        if conversation_chain:
                            st.session_state.conversation = conversation_chain
                            st.session_state.processed_docs = True
                            st.success("✅ Processing complete! You can now ask questions.")
                        else:
                            st.error("❌ Failed to create conversation chain. Check your API token.")
                            
                    except Exception as e:
                        st.error(f"❌ Error during processing: {str(e)}")
                        st.exception(e)
        
        # Clear chat button
        st.divider()
        if st.button("🗑️ Clear Chat History"):
            clear_chat()
            st.success("✅ Chat history cleared!")
            st.rerun()
        
        # Info section
        st.divider()
        st.markdown("### ℹ️ How to Use")
        st.markdown("""
        1. Upload one or more PDF files
        2. Click **Process Documents**
        3. Wait for processing to complete
        4. Ask questions about your documents
        5. Get AI-powered answers!
        """)
        
        # Setup instructions
        with st.expander("⚙️ Setup Instructions"):
            st.markdown("""
            **Required Environment Variable:**
            
            Create a `.env` file with:
            ```
            HF_TOKEN=your_huggingface_token_here
            ```
            
            Get your free token from:
            [HuggingFace Settings](https://huggingface.co/settings/tokens)
            
            **Install Dependencies:**
            ```bash
            pip install streamlit python-dotenv PyPDF2 
            pip install langchain langchain-huggingface 
            pip install langchain-community faiss-cpu 
            pip install huggingface-hub sentence-transformers
            ```
            """)

    # Main chat interface
    st.divider()
    
    if st.session_state.processed_docs:
        st.success("✅ Documents processed - Ready to answer questions!")
    else:
        st.info("👈 Please upload and process your PDF documents to get started")
    
    # Chat input
    user_question = st.chat_input("Ask a question about your documents...")
    
    if user_question:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()