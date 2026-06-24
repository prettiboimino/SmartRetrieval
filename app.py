import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
import streamlit_authenticator as stauth
import os

# 1. Page Configuration
st.set_page_config(page_title="SmartRetrieval", page_icon="📄", layout="wide")

# 2. User Database (Hardcoded for now)
names = ["Admin", "Client"]
usernames = ["admin", "client"]
passwords = ["admin123", "client123"] # In real life, these are hashed, but this is fine for your MVP demo!
authenticator = stauth.Authenticate(names, usernames, passwords, "smartretrieval_cookie", "abcdef", cookie_expiry_days=30)

# 3. Show Login Form
name, authentication_status, username = authenticator.login("Login", "main")

# 4. What happens if they are NOT logged in
if authentication_status == False:
    st.error("Username/password is incorrect")
elif authentication_status == None:
    st.warning("Please enter your username and password to access SmartRetrieval.")

# 5. What happens if they ARE logged in
elif authentication_status:
    # Show a logout button in the sidebar
    authenticator.logout("Logout", "sidebar")
    st.sidebar.title(f"Welcome, {name}!")

    # Hide Streamlit default menus
    hide_st_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
    """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    # Main Chat Area
    st.title("SmartRetrieval 📄")
    st.write("Your intelligent assistant for document storage and retrieval.")

    # File Uploader
    uploaded_files = st.file_uploader("📁 Upload one or more PDF documents here", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
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
            
        st.divider()
        st.subheader("Ask a Question")
        st.success(f"{len(uploaded_files)} document(s) loaded! You can now ask questions.")
        
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
                    
                with st.chat_message("assistant"):
                    st.write(response.content)
    else:
        st.info("Please upload one or more PDF documents to begin.")
