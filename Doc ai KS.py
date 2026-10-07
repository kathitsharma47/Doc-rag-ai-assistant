import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Doc RAG Assistant", page_icon="📄", layout="wide")
st.title("📄 Multimodal Technical Document RAG Assistant")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter Groq API Key:", type="password", placeholder="gsk_...")
    st.markdown("[Get a free Groq API key here](https://console.groq.com/keys)")
    
    st.markdown("---")
    st.markdown("### Model Details")
    st.markdown("- **LLM:** Llama 3.1 8B Instant")
    st.markdown("- **Embeddings:** all-MiniLM-L6-v2 (Local/Free)")

# Initialize session state variables
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

uploaded_file = st.file_uploader("Upload a PDF syllabus, paper, or research notes", type=["pdf"])

# Step 1: Process and Vectorize PDF (Runs ONCE per file upload)
if uploaded_file and api_key:
    if st.session_state.processed_file != uploaded_file.name:
        with st.status("Processing document...", expanded=True) as status:
            st.write("Extracting text from PDF...")
            reader = PdfReader(uploaded_file)
            raw_text = ""
            for i, page in enumerate(reader.pages):
                extracted = page.extract_text()
                if extracted:
                    raw_text += extracted + "\n"

            if not raw_text.strip():
                st.error("Error: Could not extract text from this PDF. It might be an image-only scan.")
                st.stop()

            st.write(f"Chunking document text ({len(reader.pages)} pages parsed)...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            docs = text_splitter.create_documents([raw_text])

            st.write("Generating vector embeddings (in-memory FAISS)...")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(docs, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            st.write("Initializing LLaMA 3.1 RAG chain...")
            llm = ChatGroq(
                temperature=0.1, 
                groq_api_key=api_key, 
                model_name="llama-3.1-8b-instant"
            )

            system_prompt = (
                "You are an expert academic assistant. Answer the user's question strictly "
                "based on the retrieved context below. If the answer cannot be found, state "
                "truthfully that the document does not contain this information.\n\n"
                "Context:\n{context}"
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}")
            ])

            qa_chain = create_stuff_documents_chain(llm, prompt)
            st.session_state.rag_chain = create_retrieval_chain(retriever, qa_chain)
            st.session_state.processed_file = uploaded_file.name

            status.update(label="Document indexed and ready for questions!", state="complete", expanded=False)

# Step 2: Query Interface
if st.session_state.rag_chain:
    user_query = st.text_input("Ask any question regarding your uploaded document:", placeholder="e.g., What are the core findings of Section 3?")
    
    if user_query:
        with st.spinner("Analyzing document chunks..."):
            try:
                response = st.session_state.rag_chain.invoke({"input": user_query})
                
                st.markdown("### 💡 Response")
                st.write(response["answer"])

                with st.expander("🔍 View Retrieved Source Chunks"):
                    for idx, doc in enumerate(response["context"]):
                        st.markdown(f"**Chunk {idx+1}:**")
                        st.caption(doc.page_content)
            except Exception as e:
                st.error(f"Error executing query: {str(e)}")

elif not uploaded_file:
    st.info("👆 Please upload a PDF file to begin.")
elif not api_key:
    st.warning("👈 Please enter your Groq API Key in the sidebar.")
