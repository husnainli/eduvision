import os
import tempfile
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer, util


# -----------------------------------
# 🔁 Cached Model Loaders
# -----------------------------------
@st.cache_resource
def get_hf_embedding_model(model_name='intfloat/multilingual-e5-base'):
    return HuggingFaceEmbeddings(model_name=model_name)

@st.cache_resource
def get_sentence_transformer(model_name='thenlper/gte-small'):
    return SentenceTransformer(model_name)


# -----------------------------------
# 🧩 Chunking Arabic Text
# -----------------------------------
def chunk_text(text, chunk_size=500, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n", ".", "،", " "],
        length_function=len,
    )
    return splitter.split_text(text)


# -----------------------------------
# 📦 Embed + Cache FAISS Index
# -----------------------------------
def embed_chunks(chunks, filename, embedding_model, text_hash):
    temp_dir = tempfile.gettempdir()
    index_path = os.path.join(temp_dir, f"faiss_{filename}_{text_hash}")

    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)

    texts_with_metadata = [{"page_content": chunk, "metadata": {"source": filename}} for chunk in chunks]
    texts = [item["page_content"] for item in texts_with_metadata]
    metadatas = [item["metadata"] for item in texts_with_metadata]

    vectorstore = FAISS.from_texts(texts, embedding_model, metadatas=metadatas)
    vectorstore.save_local(index_path)
    return vectorstore


# -----------------------------------
# 🧠 Retrieve from FAISS
# -----------------------------------
def retrieve_similar_chunks(vectorstore, query, k=6):
    return vectorstore.similarity_search(query, k=k)


def retrieve_from_all_vectorstores(query, vectorstores, k_per_doc=3):
    all_matches = []
    for filename, vs in vectorstores:
        results = vs.similarity_search(query, k=k_per_doc)
        for doc in results:
            doc.metadata["source"] = filename
            all_matches.append(doc)
    return all_matches


# -----------------------------------
# 🔍 Summary Finder
# -----------------------------------
def find_most_similar_summary(response, summaries, model=None):
    if model is None:
        model = get_sentence_transformer()

    summary_texts = [s for (_, s) in summaries]
    summary_embeddings = model.encode(summary_texts, convert_to_tensor=True)
    response_embedding = model.encode(response, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(response_embedding, summary_embeddings)[0]
    best_idx = similarities.argmax().item()
    return summaries[best_idx][0]  # Return filename
