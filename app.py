import streamlit as st
import fitz  # PyMuPDF
import re
import hashlib
from utils.embeddings import chunk_text, embed_chunks, retrieve_from_all_vectorstores, find_most_similar_summary
from utils.llm import query_llama3, summarize_text_arabic
from utils.translate import translate_text

# -------------------------------
# 🧼 Arabic Text Cleaning Utility
# -------------------------------
def clean_arabic_text(text):
    """Cleans and normalizes Arabic text."""
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
    """Extracts and cleans Arabic text from uploaded PDF."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    raw_text = ''.join(page.get_text() for page in doc)
    return clean_arabic_text(raw_text)

# -------------------------------
# 🚀 Streamlit App Initialization
# -------------------------------
st.set_page_config(page_title="📚 EduVision AI", layout="wide")
st.title("🤖 EduVision AI")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------
# 📤 PDF Upload
# -------------------------------
uploaded_files = st.file_uploader("📤 Upload Arabic PDFs", type=["pdf"], accept_multiple_files=True)

vectorstores = []

if uploaded_files:
    summaries = []

    for pdf_file in uploaded_files:
        filename = pdf_file.name
        st.success(f"✅ {filename} uploaded successfully!")

        # 🔍 Extract and clean text
        with st.spinner(f"🧼 Extracting and cleaning text from {filename}..."):
            pdf_text = extract_text_from_pdf(pdf_file)

        # 🔄 Split into chunks
        with st.spinner(f"🔄 Splitting {filename} into chunks..."):
            chunks = chunk_text(pdf_text)

        # 🧠 Hash PDF text to cache vector store
        text_hash = compute_text_hash(pdf_text)

        # 🧠 Generate and cache embeddings
        with st.spinner(f"🧠 Embedding text from {filename}..."):
            vs = get_cached_vectorstore(chunks, filename, text_hash)
            vectorstores.append((filename, vs))

        # 📝 Generate summary of full text
        with st.spinner(f"📚 Generating Arabic summary for {filename}..."):
            summary = get_cached_summary(pdf_text, filename, text_hash)
            summaries.append((filename, summary))

        with st.container():
            st.markdown(f"### 📝 ملخص الوثيقة: {filename}")
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

    # ---------------------------------
    # 💬 Interactive Q&A Chat Interface
    # ---------------------------------
    st.divider()
    st.subheader("💬 Ask a question based on the uploaded PDFs")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("✍️ اكتب سؤالك هنا (باللغة العربية)...")

    if user_input:
        # Show user question
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Retrieve relevant document chunks
        with st.spinner("🤖 Generating response using LLaMA 3..."):
            retrieved_docs = retrieve_from_all_vectorstores(vectorstores, user_input, k_per_doc=4)

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

            # # ✅ Extract unique source filenames
            # used_sources = set(doc.metadata.get("source", "غير معروف") for doc in retrieved_docs)
            # sources_line = "🗂️ المراجع: [" + "، ".join(used_sources) + "]"

            # from collections import Counter
            # source_counter = Counter(doc.metadata.get("source", "غير معروف") for doc in retrieved_docs)
            # top_sources = [src for src, count in source_counter.most_common(1)]
            # sources_line = "🗂️ المراجع: [" + "، ".join(top_sources) + "]"
            
            # ✅ NEW: Match response to the most similar summary
            top_source = find_most_similar_summary(response, summaries)
            sources_line = f"🗂️ المرجع: [{top_source}]"
            # ✅ Append source references to the final message
            final_response = f"{response.strip()}\n\n{sources_line}"

        # ✅ Show the response
        with st.chat_message("assistant"):
            st.markdown(final_response)

        with st.spinner("Translating to English..."):
            translation_result = translate_text(final_response)
            st.markdown(f"📗 English Translation: `{translation_result}`")

        # ✅ Add to chat history
        st.session_state.messages.append({"role": "assistant", "content": final_response})

else:
    st.info("⬆️ الرجاء رفع ملف PDF لبدء المحادثة.")
