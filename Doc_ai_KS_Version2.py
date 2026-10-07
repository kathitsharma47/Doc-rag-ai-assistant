import streamlit as st
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Doc RAG Assistant", page_icon="📄", layout="wide")
st.title("📄 Technical Document RAG Assistant")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter Groq API Key:", type="password", placeholder="gsk_...")
    st.markdown("[Get a free Groq API key here](https://console.groq.com/keys)")

    st.markdown("---")
    st.markdown("### Model Stack")
    st.markdown("- **Engine:** PyMuPDF (`fitz`)")
    st.markdown("- **LLM:** LLaMA 3.1 8B Instant")
    st.markdown("- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2")

# Initialize session state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

uploaded_file = st.file_uploader("Upload a PDF document, syllabus, or paper", type=["pdf"])

# Step 1: Parse and Vectorize PDF using PyMuPDF
if uploaded_file and api_key:
    if st.session_state.processed_file != uploaded_file.name:
        with st.status("Processing document...", expanded=True) as status:
            st.write("Reading uploaded PDF...")

            file_bytes = uploaded_file.read()
            if not file_bytes:
                st.error("The uploaded file is empty.")
                st.stop()

            try:
                pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
            except Exception as e:
                st.error(f"Unable to open PDF: {e}")
                st.stop()

            if pdf_document.is_encrypted:
                try:
                    pdf_document.authenticate("")
                except Exception:
                    st.error("This PDF is encrypted and cannot be read without a password.")
                    st.stop()

            raw_text = ""
            total_pages = len(pdf_document)
            for page_num in range(total_pages):
                page = pdf_document[page_num]
                raw_text += page.get_text() + "\n"

            pdf_document.close()

            if not raw_text.strip():
                st.error("Error: No extractable text detected. This file might be an image-only scan.")
                st.stop()

            st.write(f"Chunking document text ({total_pages} pages parsed)...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            docs = text_splitter.create_documents([raw_text])

            st.write("Computing vector embeddings...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            vectorstore = FAISS.from_documents(docs, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            st.write("Setting up LLaMA 3.1 inference pipeline...")
            llm = ChatGroq(
                temperature=0.1,
                groq_api_key=api_key,
                model="llama-3.1-8b-instant"
            )

            system_prompt = """
            You are an academic and technical research assistant.
            Answer the user's question strictly using the retrieved document context below.
            If the context does not contain the necessary information, say:
            "The document does not mention this information."
            Do not invent facts.

            Context:
            {context}
            """

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}")
            ])

            qa_chain = create_stuff_documents_chain(llm, prompt)
            st.session_state.rag_chain = create_retrieval_chain(retriever, qa_chain)
            st.session_state.processed_file = uploaded_file.name

            status.update(
                label="Document indexed successfully! You can now query it.",
                state="complete",
                expanded=False
            )

# Step 2: Query Interface
if st.session_state.rag_chain:
    user_query = st.text_input(
        "Ask a question about the document:",
        placeholder="e.g., What are the key findings or algorithms discussed?"
    )

    if user_query:
        with st.spinner("Generating answer from context..."):
            try:
                response = st.session_state.rag_chain.invoke({"input": user_query})

                st.markdown("### 💡 Response")
                st.write(response.get("answer", "No answer generated."))

                if "context" in response and response["context"]:
                    with st.expander("🔍 View Retrieved Document Sources"):
                        for idx, doc in enumerate(response["context"]):
                            st.markdown(f"**Chunk {idx+1}:**")
                            st.caption(doc.page_content)

            except Exception as e:
                st.error(f"Error querying model: {str(e)}")

elif not uploaded_file:
    st.info("👆 Upload a PDF file above to begin.")
elif not api_key:
    st.warning("👈 Enter your Groq API key in the sidebar.")