from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import tempfile
import pickle
import os

def chunk_text(text, chunk_size=500, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n", ".", "،", " "]
    )
    return splitter.split_text(text)


def embed_chunks(chunks, model_name='intfloat/multilingual-e5-base'):
    embedding = HuggingFaceEmbeddings(model_name=model_name)

    # Use a temp directory for FAISS index persistence
    temp_dir = tempfile.gettempdir()
    index_path = os.path.join(temp_dir, "faiss_index")

    # If index already exists, load it
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embedding, allow_dangerous_deserialization=True)

    # Otherwise, create and save it
    vectorstore = FAISS.from_texts(chunks, embedding)
    vectorstore.save_local(index_path)
    return vectorstore


def retrieve_similar_chunks(vectorstore, query, k=6):
    return vectorstore.similarity_search(query, k=k)
