import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatopenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore
def get_conversation_chain(vectorstore):
    llm= ChatopenAI()
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain=ConversationalRetrievalChain.formllm(

    )

def main():
    load_dotenv()
    st.set_page_config(page_title="Talk to PDF", page_icon=":books:")
    st.header("Talk to PDF :books:")
    user_question = st.text_input("Ask your question about the document:")

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on process", accept_multiple_files=True)
        if st.button("Process"):
            if not pdf_docs:
                st.error("Please upload PDF documents before processing.")
            else:
                with st.spinner("Processing PDFs..."):
                    try:
                        # Get the PDF text
                        raw_text = get_pdf_text(pdf_docs)
                        
                        # Get the text chunks
                        text_chunks = get_text_chunks(raw_text)
                        st.write(f"Created {len(text_chunks)} text chunks.")
                        
                        # Create vector store
                        vectorstore = get_vectorstore(text_chunks)
                        st.success("Processing complete! Vector store created successfully.")
                    except Exception as e:
                        st.error(f"An error occurred during processing: {str(e)}")

                        #create conversation chain
                        conversation=get_conversation_chain(vectorstore)


    if user_question:
        st.write("You asked:", user_question)
        # Here you would typically query the vectorstore and display results
        # This part is not implemented in your current code

if __name__ == '__main__':
    main()