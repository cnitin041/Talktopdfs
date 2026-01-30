import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import re

# CSS styles
css = '''
<style>
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex
}

.chat-message.user {
    background-color: #2b313e
}

.chat-message.bot {
    background-color: #475063
}

.chat-message .avatar {
    width: 20%;
    display: flex;
    align-items: center;
    justify-content: center;
}

.chat-message .avatar i {
    font-size: 3.5rem;
    color: #fff;
}

.chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
    color: #fff;
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
    llm = HuggingFaceEndpoint(repo_id="google/flan-t5-base", temperature=0.3, max_length=512)
    
    template = """Use the following context to answer the question. If you don't know the answer, say you don't know.

Context: {context}

Question: {question}

Answer:"""
    
    prompt = PromptTemplate.from_template(template)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def is_greeting(text):
    greetings = r"\b(hi|hello|hey|greetings|good morning|good afternoon|good evening|hi, how are you)\b"
    return bool(re.search(greetings, text.lower()))

def handle_greeting(greeting):
    responses = {
        "hi": "Hi there! How can I help you with your documents today?",
        "hello": "Hello! I'm ready to assist you with any questions about your PDFs.",
        "hey": "Hey! What would you like to know about your documents?",
        "greetings": "Greetings! I'm here to help you with your PDF queries.",
        "good morning": "Good morning! How may I assist you with your documents today?",
        "good afternoon": "Good afternoon! What questions do you have about your PDFs?",
        "good evening": "Good evening! I'm here to help with any document-related questions."
    }
    return responses.get(greeting.lower(), "Hello! How can I assist you with your documents today?")

def handle_userinput(user_question):
    if is_greeting(user_question):
        st.write(bot_template.replace("{{MSG}}", handle_greeting(user_question)), unsafe_allow_html=True)
        return

    if "conversation" not in st.session_state or st.session_state.conversation is None:
        st.error("Please process your documents before asking questions.")
        return

    # Initialize chat history if not exists
    if st.session_state.chat_history is None:
        st.session_state.chat_history = []

    # Get response from chain (returns string directly)
    answer = st.session_state.conversation.invoke(user_question)
    
    # Update chat history
    st.session_state.chat_history.append((user_question, answer))

    # Display all messages
    for question, ans in st.session_state.chat_history:
        st.write(user_template.replace("{{MSG}}", question), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", ans), unsafe_allow_html=True)

def clear_chat():
    st.session_state.chat_history = []
    st.session_state.conversation = None

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with multiple PDFs :books:")
    
    # Sidebar
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if not pdf_docs:
                st.error("Please upload PDF documents before processing.")
            else:
                with st.spinner("Processing"):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        vectorstore = get_vectorstore(text_chunks)
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        st.success("Processing complete! You can now ask questions about your documents.")
                    except Exception as e:
                        st.error(f"An error occurred during processing: {str(e)}")
        
        if st.button("Clear Chat"):
            clear_chat()
            st.success("Chat cleared!")

    # Main chat interface
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()