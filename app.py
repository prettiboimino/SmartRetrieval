import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
import os

# 1. Page Configuration
st.set_page_config(page_title="SmartRetrieval", page_icon="📄", layout="wide")

# 2. Hide Streamlit default menus
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# 3. Main Chat Area
st.title("SmartRetrieval 📄")
st.write("Your intelligent assistant for document storage and retrieval.")

# 4. File Uploader (NOW ACCEPTS MULTIPLE FILES!)
uploaded_files = st.file_uploader("📁 Upload one or more PDF documents here", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    # Load and process the PDFs
    with st.spinner("AI is reading your document(s)..."):
        all_docs = []
        
        # Loop through every file uploaded
        for uploaded_file in uploaded_files:
            # Save the file temporarily
            temp_path = "temp_" + uploaded_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Read the PDF and add it to our list
            loader = PyPDFLoader(temp_path)
            all_docs.extend(loader.load())
            
            # Clean up the temp file
            os.remove(temp_path)
        
        # Split ALL the combined text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(all_docs)
        
        # USE FASTEMBED
        embeddings = FastEmbedEmbeddings()
        
        # USE IN-MEMORY VECTOR STORE (Now holds all the files!)
        vectorstore = InMemoryVectorStore.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()
        
        # PULL GROQ KEY SECURELY FROM STREAMLIT SECRETS
        groq_api_key = st.secrets["GROQ_API_KEY"]
        llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)
        
    st.divider()
    st.subheader("Ask a Question")
    st.success(f"{len(uploaded_files)} document(s) loaded! You can now ask questions.")
    
    # 5. Chat Input box
    question = st.chat_input("Ask a question about your document(s)...")
    
    if question:
        with st.spinner("Thinking..."):
            # Get the relevant chunks
            retrieved_docs = retriever.invoke(question)
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # Create the prompt
            prompt = f"""You are a helpful assistant for a Ghanaian business. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, say that you don't know. 

Context:
{context_text}

Question: {question}
Answer:"""
            
            # Get AI response and display it like a chat
            response = llm.invoke(prompt)
            
            with st.chat_message("user"):
                st.write(question)
                
            with st.chat_message("assistant"):
                st.write(response.content)
else:
    # What to show when no document is uploaded yet
    st.info("Please upload one or more PDF documents to begin.")
