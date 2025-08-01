import streamlit as st
import fitz  # PyMuPDF
import re
import hashlib

from utils.embeddings import chunk_text, embed_chunks, retrieve_from_all_vectorstores, find_most_similar_summary
from utils.llm import query_llama3, summarize_text_arabic
from utils.translate import translate_text

import os
import gc
import tempfile

# -------------------------------
# 🧼 Arabic Text Cleaning Utility
# -------------------------------
def clean_arabic_text(text):
    diacritics = re.compile(r"[ًٌٍَُِّْٓ]")
    text = re.sub(diacritics, '', text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[^\u0600-\u06FF\s]", '', text)
    return re.sub(r"\s+", ' ', text).strip()

# -------------------------------
# 🧠 Hashing for Caching
# -------------------------------
def compute_text_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

@st.cache_resource
def get_cached_vectorstore(chunks, filename, text_hash):
    return embed_chunks(chunks, filename=filename, text_hash=text_hash)

@st.cache_data(show_spinner=False)
def get_cached_summary(text, filename, text_hash):
    return summarize_text_arabic(text)

# -------------------------------
# 📄 PDF Text Extraction
# -------------------------------
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    raw_text = ''.join(page.get_text() for page in doc)
    return clean_arabic_text(raw_text)

# -------------------------------
# 🚀 Streamlit App Initialization
# -------------------------------
st.set_page_config(page_title="📚 EduVision AI", layout="wide")
st.title("🤖 EduVision AI")


if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------
# 📤 PDF Upload
# -------------------------------
uploaded_files = st.file_uploader("📤 Upload up to 3 Arabic PDFs (Max 5MB each)", type=["pdf"], accept_multiple_files=True)
vectorstores = []

# ✅ Enforce limits on number, size, and duplicates
if uploaded_files:
    # Limit number of files to 3
    if len(uploaded_files) > 3:
        st.warning("⚠️ You can upload a maximum of 3 PDF files at a time.")
        uploaded_files = uploaded_files[:3]  # Only take the first 3

    valid_files = []
    seen_hashes = set()
    seen_names = set()

    for f in uploaded_files:
        file_size_mb = f.size / (1024 * 1024)

        # Compute file hash (for exact duplicate detection)
        file_bytes = f.read()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        f.seek(0)  # Reset file pointer after reading

        if f.name in seen_names:
            st.warning(f"⚠️ Duplicate file name detected: {f.name}")
            continue
        if file_hash in seen_hashes:
            st.warning(f"⚠️ Duplicate file content detected: {f.name}")
            continue
        if file_size_mb > 5:
            st.warning(f"❌ {f.name} is too large ({file_size_mb:.2f} MB). Max allowed: 5 MB.")
            continue

        # Passed all checks
        valid_files.append(f)
        seen_hashes.add(file_hash)
        seen_names.add(f.name)

    if not valid_files:
        st.stop()

    uploaded_files = valid_files  # Only valid, unique, small files

if uploaded_files:
    summaries = []
    vectorstores = []

    # UI container to log progress
    processing_status = st.empty()
    progress_container = st.container()
    summary_container = st.container()

    processing_status.info("🚀 Processing PDF file(s)...")

    for i, pdf_file in enumerate(uploaded_files, start=1):
        filename = pdf_file.name
        try:
            pdf_text = extract_text_from_pdf(pdf_file)
            text_hash = compute_text_hash(pdf_text)
            index_path = os.path.join(tempfile.gettempdir(), f"faiss_index_{filename}_{text_hash}")

            # Only chunk if vectorstore doesn't already exist
            if not os.path.exists(index_path):
                chunks = chunk_text(pdf_text)
            else:
                chunks = []  # dummy value, not used when already cached

            vs = get_cached_vectorstore(chunks, filename, text_hash)
            short_text = pdf_text[:5000]
            summary = get_cached_summary(pdf_text, filename, text_hash)

            vectorstores.append((filename, vs))
            summaries.append((filename, summary))

            with progress_container:
                st.success(f"✅ {filename} processed ({i}/{len(uploaded_files)})")

            # 🧹 Clean up memory for large objects
            del pdf_text
            del chunks
            del vs
            del summary
            gc.collect()

        except Exception as e:
            with progress_container:
                st.error(f"❌ Failed to process {filename}: {str(e)}")

    processing_status.success("✅ All PDFs processed successfully!")

    with st.sidebar.expander("🧾 جميع الملخصات", expanded=False):
        for filename, summary in summaries:
            st.markdown(
                f"""
                <div style='background-color:#f0f0f0;
                            border-left: 4px solid #1E90FF;
                            padding: 0.7rem;
                            margin-bottom: 1rem;
                            border-radius: 8px;
                            font-size: 0.95rem;
                            direction: rtl;
                            text-align: right;'>
                    <b>{filename}</b><br><br>
                    {summary}
                </div>
                """,
                unsafe_allow_html=True
            )

    
    # -------------------------------
    # 💬 Interactive Q&A Chat Interface
    # -------------------------------
    st.divider()
    st.subheader("💬 Ask a question based on the uploaded PDFs")

    with st.expander("⚙️ Chat Options", expanded=False):

        col1, col2 = st.columns([1.5, 1])

        with col1:
            st.markdown("<br>", unsafe_allow_html=True)  # spacing
            if st.button("Clear Chat History", help="Reset the conversation."):
                st.session_state.messages = []
                st.rerun()
        
        with col2:
            ask_mode = st.radio(
                "📌 Select Question Mode:",
                ["Ask from all PDFs", "Ask from a specific PDF"],
                horizontal=True,
                index=0,
                help="Choose whether to query all uploaded documents or just one."
            )

            if ask_mode == "Ask from a specific PDF":
                selected_pdf = st.selectbox(
                    "📄 Select a PDF to query:",
                    [name for name, _ in vectorstores],
                    help="Choose a specific document for your question."
                )
                selected_vs = next(vs for name, vs in vectorstores if name == selected_pdf)


    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("✍️ اكتب سؤالك هنا (باللغة العربية)...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("🤖 Generating response using LLaMA 3..."):
            # retrieved_docs = retrieve_from_all_vectorstores(vectorstores, user_input, k_per_doc=4)
            if ask_mode == "Ask from all PDFs":
                retrieved_docs = retrieve_from_all_vectorstores(vectorstores, user_input, k_per_doc=4)
            else:
                retrieved_docs = retrieve_from_all_vectorstores([(selected_pdf, selected_vs)], user_input, k_per_doc=6)

            context = "\n\n".join(
                f"[من الملف: {doc.metadata.get('source', 'غير معروف')}]\n{doc.page_content}"
                for doc in retrieved_docs
            )

            prompt = (
                f"السؤال:\n{user_input}\n\n"
                f"محتوى الوثائق:\n{context}\n\n"
                "أجب إجابة كاملة وشاملة مستندًا فقط إلى محتوى الوثائق أعلاه.\n"
                "يجب أن تكون إجابتك باللغة العربية الفصحى فقط دون أي كلمات إنجليزية أو لغات أخرى.\n"
                "إذا لم يكن هناك معلومات كافية، قل ذلك بالعربية فقط دون تأليف."
            )

            response = query_llama3(prompt)

            top_source = find_most_similar_summary(response, summaries)
            sources_line = f"🗂️ المرجع: [{top_source}]"
            final_response = f"{response.strip()}\n\n{sources_line}"

        with st.chat_message("assistant"):
            st.markdown(final_response)

            with st.spinner("Translating to English..."):
                translation_result = translate_text(final_response)

            with st.expander("📖 Show English Translation", expanded=False):
                st.markdown(
                    f"""
                    <div style='background-color:#e3f2fd;
                                border-left: 6px solid #1976D2;
                                padding: 1.2rem;
                                border-radius: 12px;
                                font-size: 1.1rem;
                                direction: ltr;
                                text-align: left;
                                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
                                transition: all 0.3s ease;'>
                        <b>Translation:</b><br><br>
                        {translation_result}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.session_state.messages.append({"role": "assistant", "content": final_response})

else:
    st.info("⬆️ الرجاء رفع ملف PDF لبدء المحادثة.")

