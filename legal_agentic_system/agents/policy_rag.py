import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def get_policy_retriever(db_path: str = "policy_faiss_index"):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    if os.path.exists(db_path):
        return FAISS.load_local(
            db_path,
            embeddings,
            allow_dangerous_deserialization=True,
        ).as_retriever(search_kwargs={"k": 4})

    docs_dir = "policy_docs"
    if not os.path.exists(docs_dir):
        return None

    loader = DirectoryLoader(docs_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    if not docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    splits = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(db_path)

    return vectorstore.as_retriever(search_kwargs={"k": 4})


policy_retriever = get_policy_retriever()
