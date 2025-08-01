import streamlit as st
import fitz  # PyMuPDF
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.embeddings import chunk_text, embed_chunks, retrieve_from_all_vectorstores, find_most_similar_summary
from utils.llm import query_llama3, summarize_text_arabic
from utils.translate import translate_text

import psutil
import os
import shutil

# RAM usage
process = psutil.Process(os.getpid())
ram_usage = process.memory_info().rss / (1024 * 1024)  # in MB

# Disk usage
total, used, free = shutil.disk_usage("/")
disk_used_mb = used / (1024 * 1024)  # in MB


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
    return embed_chunks(chunks, filename=filename)

@st.cache_data(show_spinner=False)
def get_cached_summary(text, filename, text_hash):
    short_text = text[:3000]
    return summarize_text_arabic(short_text)

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

# Sidebar display
with st.sidebar:
    st.markdown("### 🧠 Resource Monitor")
    st.markdown(f"**RAM Usage:** {ram_usage:.2f} MB")
    st.markdown(f"**Disk Usage:** {disk_used_mb:.2f} MB")
    st.markdown("---")

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
    
    # UI container to log progress
    processing_status = st.empty()
    progress_container = st.container()
    summary_container = st.container()

    processing_status.info(f"🚀 Processing PDF file(s)...")

    # Function to process a single PDF
    def process_pdf(pdf_file):
        filename = pdf_file.name
        pdf_text = extract_text_from_pdf(pdf_file)
        chunks = chunk_text(pdf_text)
        text_hash = compute_text_hash(pdf_text)
        vs = get_cached_vectorstore(chunks, filename, text_hash)
        summary = get_cached_summary(pdf_text, filename, text_hash)
        return (filename, vs, summary)

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_pdf, f): f.name for f in uploaded_files}

        for i, future in enumerate(as_completed(futures), start=1):
            filename = futures[future]
            try:
                filename, vs, summary = future.result()
                vectorstores.append((filename, vs))
                summaries.append((filename, summary))

                with progress_container:
                    st.success(f"✅ {filename} processed ({i}/{len(uploaded_files)})")

                with summary_container:
                    with st.expander(f"📝 ملخص الوثيقة: {filename}", expanded=True):
                        st.markdown(
                            f"""
                            <div style='background-color:#f9f9f9;
                                        border-left: 5px solid #4CAF50;
                                        padding: 1rem;
                                        border-radius: 10px;
                                        font-size: 1.1rem;
                                        direction: rtl;
                                        text-align: right;'>
                                {summary}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            except Exception as e:
                with progress_container:
                    st.error(f"❌ Failed to process {filename}: {str(e)}")

        processing_status.success("✅ All PDFs processed successfully!")

    
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

