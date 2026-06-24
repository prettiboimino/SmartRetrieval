import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq

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

# 3. Sidebar for uploading documents
with st.sidebar:
    st.title("📁 Document Upload")
    st.write("Upload your PDF below. The AI will read and store it securely.")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded_file:
        st.success("Document Ready!")

# 4. Main Chat Area
st.title("SmartRetrieval 📄")
st.write("Your intelligent assistant for document storage and retrieval.")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load and process the PDF
    with st.spinner("AI is reading your document..."):
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # USE FASTEMBED (No API key needed, ultra-stable on Streamlit!)
        embeddings = FastEmbedEmbeddings()
        
        # USE IN-MEMORY VECTOR STORE
        vectorstore = InMemoryVectorStore.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()
        
        # PULL GROQ KEY SECURELY FROM STREAMLIT SECRETS
        groq_api_key = st.secrets["GROQ_API_KEY"]
        llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)
        
    st.divider()
    st.subheader("Ask a Question")
    
    # 5. Chat Input box
    question = st.chat_input("Ask a question about your document...")
    
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
    st.info("Please upload a PDF document in the sidebar to begin.")
