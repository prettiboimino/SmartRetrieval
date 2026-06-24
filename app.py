import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
import os

# 1. Page Configuration
st.set_page_config(page_title="SmartRetrieval", page_icon="📄", layout="centered", initial_sidebar_state="collapsed")

# 2. Premium UI CSS Styling
premium_style = """
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    .main .block-container { max-width: 800px; padding-top: 2rem; }
    h1 { font-weight: 800 !important; letter-spacing: -0.5px; }
    .stChatMessage { border-radius: 12px; border: 1px solid #333; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .stFileUploader { border-radius: 12px; border: 1px dashed #4a4a4a; padding: 25px; background-color: #1e1e1e; }
    .stFileUploader p, .stFileUploader span { color: #fafafa !important; }
    </style>
"""
st.markdown(premium_style, unsafe_allow_html=True)

# 3. Header Section
st.markdown("<h1 style='text-align: center;'>SmartRetrieval 📄</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Enterprise AI for Document Storage & Retrieval</p>", unsafe_allow_html=True)
st.write("")

# 4. File Uploader
uploaded_files = st.file_uploader("Upload your PDF documents to begin", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    # Load and process the PDFs
    with st.spinner("AI is reading your document(s)..."):
        all_docs = []
        for uploaded_file in uploaded_files:
            temp_path = "temp_" + uploaded_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader(temp_path)
            all_docs.extend(loader.load())
            os.remove(temp_path)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(all_docs)
        
        embeddings = FastEmbedEmbeddings()
        vectorstore = InMemoryVectorStore.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()
        
        groq_api_key = st.secrets["GROQ_API_KEY"]
        llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)
        
    st.success(f"✅ {len(uploaded_files)} document(s) loaded successfully!")
    st.divider()
    
    # 5. Chat Interface
    question = st.chat_input("Ask a question about your document(s)...")
    
    if question:
        with st.spinner("Thinking..."):
            retrieved_docs = retriever.invoke(question)
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            prompt = f"""You are a helpful assistant for a Ghanaian business. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, say that you don't know. 

Context:
{context_text}

Question: {question}
Answer:"""
            
            response = llm.invoke(prompt)
            
            with st.chat_message("user"):
                st.write(question)
                
            with st.chat_message("assistant", avatar="🤖"):
                st.write(response.content)
else:
    st.info("⬆️ Upload one or more PDF files above to get started.")
