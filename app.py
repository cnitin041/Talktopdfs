import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# CSS styles with professional color scheme and clean layout
css = '''
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

/* Global color variables for professional look */
:root {
    --primary-color: #1C2833; /* Dark blue-gray for titles */
    --secondary-color: #34495E; /* Slightly lighter blue-gray for text */
    --accent-color: #3498DB; /* Blue for buttons and highlights */
    --background-color: #F4F6F7; /* Light grey background */
    --surface-color: #FFFFFF; /* White for cards and containers */
    --on-surface-color: #1C2833; /* Dark text on white surfaces */
    --on-background-color: #1C2833; /* Dark text on light backgrounds */

    /* Dark Mode Colors */
    --dark-background-color: #1C2833; /* Dark mode background */
    --dark-surface-color: #2C3E50; /* Dark surface color */
    --dark-on-surface-color: #ECF0F1; /* Light text for surfaces */
    --dark-on-background-color: #ECF0F1; /* Light text for background */
}

/* Global body styles */
body {
    font-family: 'Roboto', sans-serif;
    background-color: var(--background-color);
    color: var(--on-background-color);
}

.stApp {
    background-color: var(--background-color);
}

.main-container {
    max-width: 1100px;
    margin: 0 auto;
    padding: 2rem;
    background-color: var(--surface-color);
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.chat-container {
    background-color: var(--surface-color);
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    padding: 2rem;
    margin-top: 2rem;
}

.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.chat-message.user {
    background-color: #E8F6FE;
    border-left: 6px solid var(--accent-color);
}

.chat-message.bot {
    background-color: #F0F0F0;
    border-left: 6px solid var(--primary-color);
}

.chat-message .message {
    flex-grow: 1;
    color: var(--on-surface-color);
}

/* Text Input Styling */
.stTextInput > div > div > input {
    background-color: var(--surface-color);
    border: 1px solid var(--primary-color);
    padding: 12px;
    border-radius: 5px;
    color: var(--on-surface-color);
}

/* Button Styling */
.stButton > button {
    background-color: var(--accent-color);
    color: var(--surface-color);
    border: none;
    padding: 10px 24px;
    font-size: 16px;
    cursor: pointer;
    border-radius: 5px;
    transition: 0.4s;
}

.stButton > button:hover {
    background-color: var(--primary-color);
    color: var(--surface-color);
}

/* Header and Logo Styles */
h1, h2, h3 {
    color: var(--primary-color);
    font-weight: 700;
}

.logo-text {
    font-size: 2.5rem;
    font-weight: bold;
    color: var(--primary-color);
    text-align: center;
    margin-bottom: 1rem;
}

.sidebar .sidebar-content {
    background-color: var(--surface-color);
}

/* Upload Section Styling */
.upload-section {
    background-color: var(--surface-color);
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.chat-input {
    background-color: var(--surface-color);
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    margin-top: 20px;
}

/* Dark Mode Support */
.dark-mode body {
    background-color: var(--dark-background-color);
    color: var(--dark-on-background-color);
}

.dark-mode .stApp {
    background-color: var(--dark-background-color);
}

.dark-mode .chat-container, .dark-mode .upload-section, .dark-mode .chat-input {
    background-color: var(--dark-surface-color);
    color: var(--dark-on-surface-color);
}

.dark-mode h1, .dark-mode h2, .dark-mode h3, .dark-mode .logo-text {
    color: var(--dark-on-surface-color);
}

.dark-mode .stTextInput > div > div > input {
    background-color: var(--dark-surface-color);
    color: var(--dark-on-surface-color);
}

</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="message">{{MSG}}</div>
</div>
'''

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.5, "max_length": 512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.error("Please upload and process your documents before asking questions.")
        return

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def clear_chat():
    st.session_state.chat_history = None
    st.session_state.conversation = None

def main():
    load_dotenv()
    st.set_page_config(page_title="TalktoPDFs - Intelligent Document Interaction",
                       page_icon=":page_facing_up:",
                       layout="wide")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    # Header section
    st.markdown('<p class="logo-text">TalktoPDFs</p>', unsafe_allow_html=True)
    st.header("Intelligent Document Interaction :page_facing_up:")

    # Sidebar
    with st.sidebar:
        st.subheader("Document Management")
        with st.container():
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            pdf_docs = st.file_uploader(
                "Upload your PDFs and click 'Process'", accept_multiple_files=True)
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    if not pdf_docs:
                        st.error("Please upload PDF documents before processing.")
                    else:
                        try:
                            raw_text = get_pdf_text(pdf_docs)
                            text_chunks = get_text_chunks(raw_text)
                            vectorstore = get_vectorstore(text_chunks)
                            st.session_state.conversation = get_conversation_chain(vectorstore)
                            st.success("Documents processed successfully!")
                        except Exception as e:
                            st.error(f"An error occurred during processing: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Clear Conversation"):
            clear_chat()
            st.success("Conversation history cleared.")

    # Main chat interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown('<div class="chat-input">', unsafe_allow_html=True)
    user_question = st.text_input("Ask a question about your documents:")
    st.markdown('</div>', unsafe_allow_html=True)
    if user_question:
        handle_userinput(user_question)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### About TalktoPDFs")
    st.write("TalktoPDFs is an advanced document interaction platform that transforms the way you engage with your PDF documents. Our AI-powered system analyzes and processes your uploaded PDFs, enabling you to have natural language conversations about your documents. Simply ask questions, and receive instant, relevant answers extracted directly from your content.")

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
