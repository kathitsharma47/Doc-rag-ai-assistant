import streamlit as st
import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Doc RAG Assistant", page_icon="📄")
st.title("📄 Multimodal Technical Document RAG Assistant")

# Sidebar for Free Groq API Key
api_key = st.sidebar.text_input("Enter Groq API Key (Free at console.groq.com):", type="password")

uploaded_file = st.file_uploader("Upload a PDF syllabus, paper, or notes", type="pdf")

if uploaded_file and api_key:
    # 1. Extract text from PDF
    reader = PdfReader(uploaded_file)
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text() or ""
        
    st.success(f"Loaded {len(reader.pages)} pages successfully.")
    
    # 2. Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.create_documents([raw_text])
    
    # 3. Vector Embeddings (Runs locally/free in memory via HuggingFace)
    with st.spinner("Generating vector embeddings..."):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 4. LLM Setup via Groq
    llm = ChatGroq(temperature=0.1, groq_api_key=api_key, model_name="llama-3.1-8b-instant")
    
    system_prompt = (
        "You are an academic assistant. Use the following retrieved context to answer "
        "the student query. If the answer is not in context, state that you do not know.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # User Query
    user_query = st.text_input("Ask a question about your document:")
    if user_query:
        with st.spinner("Querying vector index & generating answer..."):
            response = rag_chain.invoke({"input": user_query})
            st.markdown("### Answer:")
            st.write(response["answer"])
            
            with st.expander("View Source Document Chunks"):
                for doc in response["context"]:
                    st.write(doc.page_content)