import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer, util
import tempfile
import os

@st.cache_resource
def get_hf_embedding_model(model_name='intfloat/multilingual-e5-base'):
    return HuggingFaceEmbeddings(model_name=model_name)

@st.cache_resource
def get_sentence_transformer(model_name='thenlper/gte-small'):
    return SentenceTransformer(model_name)

def chunk_text(text, chunk_size=500, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n", ".", "،", " "]
    )
    return splitter.split_text(text)


def embed_chunks(chunks, filename, text_hash):
    embedding = get_hf_embedding_model()  # Use cached model
    temp_dir = tempfile.gettempdir()
    index_path = os.path.join(temp_dir, f"faiss_index_{filename}_{text_hash}")

    texts_with_metadata = [{"page_content": chunk, "metadata": {"source": filename}} for chunk in chunks]
    texts = [item["page_content"] for item in texts_with_metadata]
    metadatas = [item["metadata"] for item in texts_with_metadata]

    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embedding, allow_dangerous_deserialization=True)

    vectorstore = FAISS.from_texts(texts, embedding, metadatas=metadatas)
    vectorstore.save_local(index_path)
    return vectorstore



def retrieve_similar_chunks(vectorstore, query, k=6):
    return vectorstore.similarity_search(query, k=k)


def retrieve_from_all_vectorstores(vectorstores, query, k_per_doc=3):
    all_matches = []
    for name, vs in vectorstores:
        results = vs.similarity_search(query, k=k_per_doc)
        for doc in results:
            doc.metadata["source"] = name
            all_matches.append(doc)
    return all_matches


def find_most_similar_summary(response, summaries):
    model = get_sentence_transformer()  # Use cached model
    response_embedding = model.encode(response, convert_to_tensor=True)
    summary_texts = [s for (_, s) in summaries]
    summary_embeddings = model.encode(summary_texts, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(response_embedding, summary_embeddings)[0]
    best_match_idx = similarities.argmax().item()
    best_filename = summaries[best_match_idx][0]
    return best_filename
