import io
import json
import os
import re
import time
import html

import pymupdf as fitz
import requests
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

CSS = """
<style>
:root {
  --bg: #212121;
  --bg-sidebar: #171717;
  --bg-elevated: #303030;
  --bg-bubble: #2f2f2f;
  --bg-hover: #2a2a2a;
  --text: #ececec;
  --text-secondary: #b4b4b4;
  --text-faint: #8a8a8a;
  --border: rgba(255, 255, 255, 0.10);
  --border-strong: rgba(255, 255, 255, 0.18);
  --chat-width: 768px;
  --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --font-mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
}

html, body, .stApp {
  background: var(--bg) !important;
  color: var(--text);
  font-family: var(--font-sans);
  -webkit-font-smoothing: antialiased;
}

[data-testid="stAppDeployButton"],
[data-testid="stStatusWidget"],
[data-testid="stToolbarActions"],
[data-testid="stMainMenuButton"],
[data-testid="stDecoration"],
#MainMenu, footer { display: none !important; }

[data-testid="stHeader"] {
  background: rgba(33, 33, 33, 0.86) !important;
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
}
[data-testid="stMain"] { background: var(--bg) !important; }

[data-testid="stMainBlockContainer"] {
  max-width: var(--chat-width) !important;
  padding-left: 1.5rem !important;
  padding-right: 1.5rem !important;
  margin: 0 auto !important;
}

[data-testid="stSidebar"] {
  background: var(--bg-sidebar) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child { background: var(--bg-sidebar) !important; }
[data-testid="stSidebarContent"] { background: var(--bg-sidebar) !important; }
[data-testid="stSidebarUserContent"] { padding: 0.5rem 0.85rem 1.25rem !important; }
[data-testid="stSidebarHeader"] { padding-bottom: 0 !important; }

h1, h2, h3, h4, h5, h6 { color: var(--text) !important; letter-spacing: -0.01em; }
[data-testid="stChatMessageContent"],
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
  font-family: var(--font-sans) !important;
}
h1, h2, h3, h4, h5, h6 { font-family: var(--font-sans) !important; }
p, li, span { color: inherit; }
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p { color: var(--text); }
[data-testid="stCaptionContainer"], .stCaption { color: var(--text-secondary) !important; }
a { color: #8ab4f8 !important; text-decoration: none; }
a:hover { text-decoration: underline; }

[data-testid="stBaseButton-secondary"] {
  background: transparent !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  font-weight: 500 !important;
  box-shadow: none !important;
  transition: background 0.15s ease, border-color 0.15s ease !important;
}
[data-testid="stBaseButton-secondary"]:hover {
  background: var(--bg-hover) !important;
  border-color: var(--border-strong) !important;
}
[data-testid="stBaseButton-primary"] {
  background: var(--text) !important;
  color: #1a1a1a !important;
  border: 1px solid transparent !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  box-shadow: none !important;
  transition: background 0.15s ease !important;
}
[data-testid="stBaseButton-primary"]:hover { background: #ffffff !important; }
[data-testid="stBaseButton-primary"] p,
[data-testid="stBaseButton-primary"] span { color: #1a1a1a !important; }
[data-testid="stBaseButton-secondary"]:disabled {
  opacity: 0.4 !important;
  cursor: not-allowed !important;
}
[data-testid="stBaseButton-primary"]:disabled {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border) !important;
  opacity: 1 !important;
  cursor: not-allowed !important;
}
[data-testid="stBaseButton-primary"]:disabled p,
[data-testid="stBaseButton-primary"]:disabled span { color: var(--text-faint) !important; }

[data-testid="stFileUploader"] { margin-bottom: 0.25rem; }
[data-testid="stFileUploaderDropzone"] {
  background: var(--bg-elevated) !important;
  border: 1px dashed var(--border-strong) !important;
  border-radius: 12px !important;
  padding: 1rem 0.85rem !important;
  min-height: 0 !important;
  transition: border-color 0.15s ease, background 0.15s ease !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--text-secondary) !important;
  background: var(--bg-hover) !important;
}
[data-testid="stFileUploaderDropzone"] * { color: var(--text-secondary) !important; }
[data-testid="stFileUploaderDropzoneInstructions"] small { color: var(--text-faint) !important; }

[data-testid="stStatus"] {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
}
[data-testid="stStatus"] * { color: var(--text) !important; }

[data-testid="stExpander"] {
  background: transparent !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
}
[data-testid="stExpander"] summary { color: var(--text-secondary) !important; }
[data-testid="stExpander"] summary:hover { color: var(--text) !important; }

[data-testid="stAlert"] { border-radius: 10px !important; }

[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  padding: 0.65rem 0 !important;
  gap: 0.75rem !important;
  align-items: flex-start !important;
}
[data-testid="stChatMessageAvatarUser"] { display: none !important; }
[data-testid="stChatMessageAvatarAssistant"] {
  display: inline-flex !important;
  align-items: center;
  justify-content: center;
  width: 28px !important;
  height: 28px !important;
  min-width: 28px !important;
  border-radius: 8px !important;
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-secondary) !important;
  margin-top: 2px;
}
[data-testid="stChatMessageAvatarAssistant"] span,
[data-testid="stChatMessageAvatarAssistant"] svg {
  color: var(--text-secondary) !important;
  width: 15px !important;
  height: 15px !important;
}

[data-testid="stChatMessageContent"] {
  width: 100%;
  padding: 0 !important;
  color: var(--text) !important;
  font-size: 0.9375rem;
  line-height: 1.7;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { justify-content: flex-end; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
  width: fit-content !important;
  flex: 0 1 auto !important;
  max-width: 78%;
  margin-left: auto;
  background: var(--bg-bubble);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 0.65rem 1.05rem !important;
}

[data-testid="stChatMessageContent"] p { margin: 0 0 0.7rem; }
[data-testid="stChatMessageContent"] p:last-child { margin-bottom: 0; }
[data-testid="stChatMessageContent"] ul,
[data-testid="stChatMessageContent"] ol { margin: 0.5rem 0; padding-left: 1.35rem; }
[data-testid="stChatMessageContent"] li { margin: 0.2rem 0; }
[data-testid="stChatMessageContent"] pre {
  background: #0f0f0f !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 0.9rem 1rem !important;
  overflow-x: auto !important;
}
[data-testid="stChatMessageContent"] code { font-family: var(--font-mono) !important; font-size: 0.85em; }
[data-testid="stChatMessageContent"] :not(pre) > code {
  background: var(--bg-elevated) !important;
  padding: 0.12rem 0.35rem !important;
  border-radius: 5px !important;
  color: var(--text) !important;
}
[data-testid="stChatMessageContent"] table { border-collapse: collapse; width: 100%; font-size: 0.85rem; }
[data-testid="stChatMessageContent"] th,
[data-testid="stChatMessageContent"] td { border: 1px solid var(--border); padding: 0.45rem 0.65rem; }
[data-testid="stChatMessageContent"] th { background: var(--bg-elevated); }
[data-testid="stChatMessageContent"] blockquote {
  border-left: 3px solid var(--border-strong);
  padding-left: 0.9rem;
  color: var(--text-secondary);
  margin: 0.6rem 0;
}

[data-testid="stBottom"] { background: var(--bg) !important; }
[data-testid="stBottomBlockContainer"] {
  max-width: 800px !important;
  padding-left: 1.5rem !important;
  padding-right: 1.5rem !important;
}
[data-testid="stChatInput"] { background: transparent !important; }
[data-testid="stChatInput"] > div:has([data-testid="stChatInputTextArea"]) {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border) !important;
  border-radius: 26px !important;
  padding: 0.45rem 0.45rem 0.45rem 1rem !important;
  gap: 0.4rem !important;
  box-shadow: 0 4px 18px rgba(0, 0, 0, 0.28) !important;
  transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}
[data-testid="stChatInput"] > div:has([data-testid="stChatInputTextArea"]):focus-within {
  border-color: var(--border-strong) !important;
  box-shadow: 0 6px 22px rgba(0, 0, 0, 0.38) !important;
}
[data-testid="stChatInput"] [data-baseweb="textarea"],
[data-testid="stChatInput"] [data-baseweb="base-input"] {
  background: transparent !important;
  border: none !important;
}
[data-testid="stChatInputTextArea"] {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  color: var(--text) !important;
  font-family: var(--font-sans) !important;
  font-size: 0.95rem !important;
  line-height: 1.5 !important;
  padding: 0.4rem 0 !important;
  min-height: 1.6rem !important;
  max-height: 180px !important;
  resize: none !important;
}
[data-testid="stChatInputTextArea"]::placeholder { color: var(--text-faint) !important; }
[data-testid="stChatInputSubmitButton"] {
  background: var(--text) !important;
  color: #1a1a1a !important;
  border-radius: 50% !important;
  width: 2rem !important;
  height: 2rem !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  transition: background 0.15s ease, opacity 0.15s ease !important;
}
[data-testid="stChatInputSubmitButton"]:hover { background: #ffffff !important; }
[data-testid="stChatInputSubmitButton"]:disabled { background: var(--bg-hover) !important; opacity: 0.6 !important; }

.brand { display: flex; align-items: center; gap: 0.55rem; padding: 0.35rem 0.3rem 0.85rem; }
.brand-mark {
  width: 28px; height: 28px; border-radius: 8px;
  background: var(--text); color: #1a1a1a;
  display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: 0.66rem; letter-spacing: 0.02em;
}
.brand-name { font-weight: 600; font-size: 0.95rem; color: var(--text); line-height: 1.2; }
.brand-sub { font-size: 0.7rem; color: var(--text-faint); line-height: 1.3; }

.side-label {
  font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--text-faint); font-weight: 600;
  margin: 0.85rem 0.25rem 0.45rem;
}

.doc-chip {
  display: flex; align-items: center; gap: 0.4rem;
  font-size: 0.78rem; color: var(--text-secondary);
  background: var(--bg-elevated); border: 1px solid var(--border);
  border-radius: 8px; padding: 0.35rem 0.55rem; margin-bottom: 0.3rem;
  overflow: hidden; white-space: nowrap; text-overflow: ellipsis;
}

.ready-pill {
  display: inline-flex; align-items: center; gap: 0.45rem;
  font-size: 0.76rem; color: #9be3b4;
  background: rgba(60, 180, 110, 0.12); border: 1px solid rgba(60, 180, 110, 0.25);
  border-radius: 999px; padding: 0.28rem 0.62rem; margin-top: 0.55rem;
}
.ready-dot { width: 7px; height: 7px; border-radius: 50%; background: #4ade80; }

.hero { text-align: center; margin: 13vh auto 0; max-width: 620px; }
.hero-mark {
  width: 46px; height: 46px; margin: 0 auto 1rem;
  border-radius: 13px; background: var(--text); color: #1a1a1a;
  display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: 0.9rem; letter-spacing: 0.02em;
}
.hero h1 { font-size: 1.9rem !important; font-weight: 600; margin: 0 0 0.5rem !important; }
.hero p { color: var(--text-secondary) !important; font-size: 0.95rem; margin: 0 auto; max-width: 440px; }

::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.14); border-radius: 6px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.24); }
::selection { background: rgba(255, 255, 255, 0.18); }

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: 0.01ms !important; transition-duration: 0.01ms !important; }
}
</style>
"""

NVIDIA_PARSE_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/nemotron-3.5-parse/parse"
NVIDIA_LLM_MODEL = "nvidia/nemotron-3.5-bolt-e4b"
NVIDIA_EMBEDDING_MODEL = "nvidia/nemotron-3.5-embed-1b"
PARSE_MAX_OUTPUT_TOKENS = 9000
PARSE_CONTROL_TOKEN_RE = re.compile(
    r"<(?:x|y)_[-\d.eE+]+>|<class_[^>]+>|<predict_[^>]+>|<output_[^>]+>"
)
EXCERPT_CITATION_RE = re.compile(r"\s*\[Excerpt\s+(\d+)[^\]]*\]")
SUPERSCRIPT_DIGITS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def get_nvidia_api_key():
    try:
        api_key = st.secrets.get("NVIDIA_API_KEY")
    except Exception:
        api_key = None
    return api_key or os.getenv("NVIDIA_API_KEY")


def parse_page_with_nemotron(image_bytes, api_key, max_output_tokens=PARSE_MAX_OUTPUT_TOKENS, retries=3):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    files = {
        "document": ("page.png", image_bytes, "image/png"),
        "options": (None, json.dumps({"decoding": {"max_output_tokens": max_output_tokens}}), "application/json"),
    }

    for attempt in range(retries):
        try:
            response = requests.post(NVIDIA_PARSE_URL, headers=headers, files=files, timeout=180)
        except requests.RequestException:
            if attempt >= retries - 1:
                raise
            time.sleep(2 ** attempt)
            continue

        if response.status_code == 429 or response.status_code >= 500:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue

        response.raise_for_status()
        result = response.json()
        text = PARSE_CONTROL_TOKEN_RE.sub("", result.get("content", "")).strip()
        return text, result.get("finish_reason", "stop")

    return "", "stop"


def parse_pdf_with_nemotron(pdf_name, pdf_bytes, api_key):
    documents = []
    incomplete_pages = 0
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    try:
        if pdf_doc.needs_pass:
            st.warning(f"{pdf_name} is password-protected. Skipping.")
            return documents, incomplete_pages

        total_pages = pdf_doc.page_count
        if total_pages == 0:
            return documents, incomplete_pages

        progress = st.progress(0.0, text=f"Parsing {pdf_name} (0/{total_pages})...")
        try:
            for page_number, page in enumerate(pdf_doc, start=1):
                try:
                    scale = min(
                        2048 / max(page.rect.height, 1),
                        1664 / max(page.rect.width, 1),
                    )
                    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False)
                    text, finish_reason = parse_page_with_nemotron(pixmap.tobytes("png"), api_key)
                    if finish_reason == "length":
                        incomplete_pages += 1
                    if text:
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": pdf_name,
                                    "page": page_number,
                                    "parser": "nemotron-3.5-parse",
                                    "finish_reason": finish_reason,
                                },
                            )
                        )
                except Exception as e:
                    st.warning(f"Error parsing {pdf_name} page {page_number}: {str(e)}")

                progress.progress(page_number / total_pages, text=f"Parsing {pdf_name} ({page_number}/{total_pages})...")
        finally:
            progress.empty()
    finally:
        pdf_doc.close()

    return documents, incomplete_pages


def extract_pdf_with_pypdf2(pdf_name, pdf_bytes):
    documents = []
    try:
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        for page_number, page in enumerate(pdf_reader.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                documents.append(
                    Document(
                        page_content=page_text,
                        metadata={"source": pdf_name, "page": page_number, "parser": "pypdf2"},
                    )
                )
    except Exception as e:
        st.warning(f"Error reading {pdf_name}: {str(e)}")
    return documents


def get_pdf_documents(pdf_docs):
    api_key = get_nvidia_api_key()
    if not api_key:
        st.warning(
            "NVIDIA_API_KEY not found. Falling back to basic PyPDF2 extraction "
            "(no OCR, table, or layout parsing)."
        )

    documents = []
    incomplete_pages = 0

    for pdf in pdf_docs:
        try:
            pdf_bytes = pdf.getvalue() if hasattr(pdf, "getvalue") else pdf.read()
        except Exception as e:
            st.warning(f"Error reading {pdf.name}: {str(e)}")
            continue

        if api_key:
            try:
                pdf_documents, incomplete = parse_pdf_with_nemotron(pdf.name, pdf_bytes, api_key)
                if pdf_documents:
                    documents.extend(pdf_documents)
                    incomplete_pages += incomplete
                else:
                    st.warning(f"Nemotron returned no text for {pdf.name}; trying PyPDF2 fallback.")
                    documents.extend(extract_pdf_with_pypdf2(pdf.name, pdf_bytes))
            except Exception as e:
                st.warning(f"Error parsing {pdf.name} with Nemotron: {str(e)}")
        else:
            documents.extend(extract_pdf_with_pypdf2(pdf.name, pdf_bytes))

    if incomplete_pages:
        st.warning(f"{incomplete_pages} page(s) reached the parse output limit; their extracted text may be incomplete.")

    return documents


def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return text_splitter.split_documents(documents)


@st.cache_resource(show_spinner=False)
def get_embeddings(api_key):
    return NVIDIAEmbeddings(model=NVIDIA_EMBEDDING_MODEL, api_key=api_key, truncate="END")


def get_vectorstore(text_chunks, api_key):
    return FAISS.from_documents(documents=text_chunks, embedding=get_embeddings(api_key))


def get_conversation_chain(vectorstore):
    api_key = get_nvidia_api_key()
    if not api_key:
        return None

    try:
        llm = ChatNVIDIA(
            model=NVIDIA_LLM_MODEL, api_key=api_key, temperature=0.3, top_p=0.95, max_completion_tokens=8192
        )

        system_message = SystemMessage(
            content=(
                "You are a helpful assistant that answers questions based only on "
                "the provided context from PDF documents. If the context does not "
                "contain the answer, say you do not have enough information."
            )
        )

        template = """Context from documents:
{context}

Question: {question}

Instructions:
- Answer the question based ONLY on the context provided above
- If the answer is not in the context, say "I don't have enough information in the documents to answer this question."
- Be concise but complete
- Use specific details from the context when possible
- Format for readability: use short paragraphs and bullet points for lists; avoid long walls of text
- Cite sources inline as [document name, p. N] using the source labels provided
- Never output raw labels such as [Excerpt 1] or [Source 1]

Answer:"""

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        def format_docs(docs):
            formatted = []
            for doc in docs:
                source = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page")
                page_text = f", page {page}" if page else ""
                formatted.append(f"[Source: {source}{page_text}]\n{doc.page_content}")
            return "\n\n".join(formatted)

        def run_chain(question, chat_history=None):
            try:
                docs = retriever.invoke(question)

                if not docs:
                    return "I couldn't find any relevant information in the documents to answer your question."

                context = format_docs(docs)

                messages = [system_message]
                if chat_history:
                    for role, content in chat_history[-6:]:
                        if role == "user":
                            messages.append(HumanMessage(content=content))
                        else:
                            messages.append(AIMessage(content=content))
                messages.append(HumanMessage(content=template.format(context=context, question=question)))

                response = llm.invoke(messages)
                answer = response.content

                if isinstance(answer, list):
                    answer = "".join(
                        part if isinstance(part, str) else part.get("text", "") for part in answer
                    )

                answer = str(answer).strip()
                return answer or "I couldn't generate an answer from the retrieved context."
            except Exception as e:
                return f"Sorry, I encountered an error: {str(e)}"

        return run_chain

    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        return None


def is_greeting(text):
    greetings = r"\b(hi|hello|hey|greetings|good morning|good afternoon|good evening)\b"
    return bool(re.search(greetings, text.lower().strip()))


def handle_greeting(greeting):
    greeting_lower = greeting.lower().strip()
    responses = {
        "hi": "Hi there! How can I help you with your documents today?",
        "hello": "Hello! I'm ready to assist you with any questions about your PDFs.",
        "hey": "Hey! What would you like to know about your documents?",
        "greetings": "Greetings! I'm here to help you with your PDF queries.",
        "good morning": "Good morning! How may I assist you with your documents today?",
        "good afternoon": "Good afternoon! What questions do you have about your PDFs?",
        "good evening": "Good evening! I'm here to help with any document-related questions.",
    }
    for key in responses:
        if key in greeting_lower:
            return responses[key]
    return "Hello! How can I assist you with your documents today?"


def prettify_answer(text):
    if not text:
        return text
    return EXCERPT_CITATION_RE.sub(
        lambda match: " " + match.group(1).translate(SUPERSCRIPT_DIGITS), text
    )


def init_state():
    defaults = {
        "conversation": None,
        "chat_history": [],
        "processed_docs": False,
        "doc_stats": {"files": 0, "pages": 0, "chunks": 0},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def new_chat():
    st.session_state.chat_history = []
    st.session_state.pop("pending_question", None)


def reset_all():
    new_chat()
    st.session_state.conversation = None
    st.session_state.processed_docs = False
    st.session_state.doc_stats = {"files": 0, "pages": 0, "chunks": 0}


def process_documents(pdf_docs):
    api_key = get_nvidia_api_key()
    if not api_key:
        st.error("NVIDIA_API_KEY is missing. Add it to your .env file.")
        return

    completed = False
    with st.status("Processing documents...", expanded=True) as status:
        st.write("Parsing PDF pages...")
        raw_documents = get_pdf_documents(pdf_docs)
        if not raw_documents:
            status.update(label="No text could be extracted", state="error")
            return

        st.write(f"Found {len(raw_documents)} page(s). Splitting into chunks...")
        text_chunks = get_text_chunks(raw_documents)
        if not text_chunks:
            status.update(label="Could not create text chunks", state="error")
            return

        st.write(f"Building the vector index from {len(text_chunks)} chunks...")
        vectorstore = get_vectorstore(text_chunks, api_key)

        st.write("Setting up the assistant...")
        conversation = get_conversation_chain(vectorstore)
        if not conversation:
            status.update(label="Failed to set up the assistant", state="error")
            return

        st.session_state.conversation = conversation
        st.session_state.processed_docs = True
        st.session_state.doc_stats = {
            "files": len(pdf_docs),
            "pages": len(raw_documents),
            "chunks": len(text_chunks),
        }
        completed = True
        status.update(
            label=f"Ready · {len(pdf_docs)} file(s) · {len(raw_documents)} pages",
            state="complete",
            expanded=False,
        )

    if completed:
        st.rerun()


def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div class="brand">
              <div class="brand-mark">PDF</div>
              <div>
                <div class="brand-name">PDF Chat</div>
                <div class="brand-sub">Nemotron 3.5 · RAG</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("New chat", icon=":material/add:", width="stretch", key="new_chat"):
            new_chat()
            st.rerun()

        st.markdown('<div class="side-label">Documents</div>', unsafe_allow_html=True)

        pdf_docs = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="pdf_uploader",
        )

        if pdf_docs:
            chips = "".join(
                f'<div class="doc-chip">{html.escape(pdf.name)}</div>' for pdf in pdf_docs
            )
            st.markdown(chips, unsafe_allow_html=True)

        if st.button(
            "Process documents",
            type="primary",
            width="stretch",
            disabled=not pdf_docs,
            key="process_docs",
        ):
            process_documents(pdf_docs)

        if st.session_state.processed_docs:
            stats = st.session_state.doc_stats
            st.markdown(
                f'<div class="ready-pill"><span class="ready-dot"></span>'
                f'Ready · {stats["files"]} file(s) · {stats["chunks"]} chunks</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="side-label">Model</div>', unsafe_allow_html=True)
        if get_nvidia_api_key():
            st.caption("Nemotron 3.5 Bolt · Parse · Embed")
        else:
            st.caption("Set NVIDIA_API_KEY in .env to enable")

        with st.expander("Settings"):
            st.caption("Parser: Nemotron 3.5 Parse")
            st.caption("Embeddings: Nemotron 3.5 Embed")
            st.caption("Model: Nemotron 3.5 Bolt")
            st.divider()
            if st.button("Clear documents", width="stretch", key="clear_docs"):
                reset_all()
                st.rerun()


def render_empty_state(ready):
    if ready:
        title = "What would you like to know?"
        subtitle = "Ask anything about the documents you've uploaded."
    else:
        title = "Chat with your PDFs"
        subtitle = "Upload one or more PDFs in the sidebar, then process them to start asking questions."

    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-mark">PDF</div>
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if ready:
        examples = [
            "Summarize these documents",
            "What are the key takeaways?",
            "List important dates and figures",
        ]
        cols = st.columns(3, gap="small")
        for col, example in zip(cols, examples):
            with col:
                if st.button(example, width="stretch", key=f"suggest_{example}"):
                    st.session_state.pending_question = example
                    st.rerun()


def render_history():
    for role, message in st.session_state.chat_history:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(message if role == "user" else prettify_answer(message))


def generate_answer(prompt):
    if is_greeting(prompt):
        return handle_greeting(prompt)
    conversation = st.session_state.conversation
    if conversation is None:
        return "Please upload and process your documents before asking questions."
    return conversation(prompt, st.session_state.chat_history[:-1])


def main():
    load_dotenv()

    st.set_page_config(
        page_title="PDF Chat",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()

    render_sidebar()

    ready = bool(st.session_state.processed_docs and st.session_state.conversation)
    history = st.session_state.chat_history

    pending = st.session_state.pop("pending_question", None)
    user_question = st.chat_input(
        "Ask anything about your documents..." if ready else "Upload and process documents to start...",
        disabled=not ready,
        key="chat_input",
    )
    prompt = pending or user_question

    if prompt:
        if ready:
            history.append(("user", prompt))
        else:
            st.toast("Upload and process your documents first.", icon="⚠️")

    if not history:
        render_empty_state(ready)
    else:
        render_history()

    if prompt and ready:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = generate_answer(prompt)
            st.markdown(prettify_answer(answer))
        history.append(("bot", answer))


if __name__ == "__main__":
    main()
