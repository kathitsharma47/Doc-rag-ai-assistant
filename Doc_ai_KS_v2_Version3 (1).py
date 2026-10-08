import streamlit as st
import pymupdf as fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ----------------------------
# App setup
# ----------------------------
st.set_page_config(page_title="Doc RAG Assistant v2", page_icon="📄", layout="wide")
st.title("📄 Document RAG Assistant v2.0")

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter Groq API Key", type="password", placeholder="gsk_...")
    st.markdown("[Get a free Groq API key](https://console.groq.com/keys)")
    st.markdown("---")
    st.markdown("### Model Stack")
    st.markdown("- **PDF Parser:** PyMuPDF")
    st.markdown("- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2")
    st.markdown("- **LLM:** LLaMA 3.1 8B Instant")

# Session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "llm" not in st.session_state:
    st.session_state.llm = None

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# ----------------------------
# Helper: build RAG pipeline
# ----------------------------
def build_rag_pipeline(pdf_bytes, api_key):
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Cannot open PDF: {str(e)}")

    if pdf_document.is_encrypted:
        try:
            pdf_document.authenticate("")
        except Exception:
            raise ValueError("This PDF is encrypted and cannot be read without a password.")

    raw_text = ""
    try:
        for page_num in range(len(pdf_document)):
            raw_text += pdf_document[page_num].get_text() + "\n"
    finally:
        pdf_document.close()

    if not raw_text.strip():
        raise ValueError("No extractable text found in the PDF.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
    )
    chunks = text_splitter.split_text(raw_text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    llm = ChatGroq(
        groq_api_key=api_key,
        model="llama-3.1-8b-instant",
        temperature=0.1
    )

    return vectorstore, llm

# ----------------------------
# Processing
# ----------------------------
if uploaded_file and api_key:
    if st.session_state.processed_file != uploaded_file.name:
        with st.spinner("Processing PDF..."):
            try:
                file_bytes = uploaded_file.read()
                vectorstore, llm = build_rag_pipeline(file_bytes, api_key)

                st.session_state.vectorstore = vectorstore
                st.session_state.llm = llm
                st.session_state.processed_file = uploaded_file.name
                st.session_state.chat_history = []

                st.success("✅ Document processed successfully. You can now ask questions.")
            except Exception as e:
                st.error(f"❌ Error processing PDF: {str(e)}")
                st.stop()

# ----------------------------
# Query UI
# ----------------------------
if st.session_state.vectorstore and st.session_state.llm and api_key:
    st.markdown("### 💬 Ask a question about the uploaded document")

    # Show chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask something about the document...")

    if user_query:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Retrieve relevant chunks
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(user_query)

        context = "\n\n".join(doc.page_content for doc in relevant_docs)

        system_prompt_template = """You are a technical research assistant.
Answer using only the document context provided below.
If the document does not contain the answer, say:
"The document does not mention this information."
Do not invent facts.

Context:
{context}

Question: {input}"""

        prompt = ChatPromptTemplate.from_template(system_prompt_template)

        chain = prompt | st.session_state.llm

        with st.spinner("🔄 Generating answer..."):
            try:
                response = chain.invoke(
                    {
                        "context": context,
                        "input": user_query
                    }
                )

                answer = response.content if hasattr(response, "content") else str(response)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

                with st.chat_message("assistant"):
                    st.markdown(answer)

                with st.expander("🔍 View retrieved sources"):
                    for idx, doc in enumerate(relevant_docs, start=1):
                        st.markdown(f"**Chunk {idx}:**")
                        st.caption(doc.page_content)

            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")

elif not uploaded_file:
    st.info("👆 Upload a PDF file to begin.")
elif not api_key:
    st.warning("👈 Enter your Groq API key in the sidebar.")
