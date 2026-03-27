import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def initialize_vector_store(pdf_path="data/indian_constitution.pdf", db_path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    if os.path.exists(db_path):
        return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Please place the constitution PDF at {pdf_path}")
        
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(db_path)
    return vectorstore

# Export a retriever instance
vectorstore = initialize_vector_store()
constitution_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
