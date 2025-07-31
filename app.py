import streamlit as st
import fitz  # PyMuPDF
import re

from utils.embeddings import chunk_text, embed_chunks, retrieve_similar_chunks
from utils.llm import query_llama3
from utils.translate import translate_text
from utils.jais_llm import query_jais

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
# 📄 PDF Text Extraction
# -------------------------------
def extract_text_from_pdf(pdf_file):
    """Extracts and cleans Arabic text from uploaded PDF."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    raw_text = ''.join(page.get_text() for page in doc)
    return clean_arabic_text(raw_text)

def sanitize_for_translation(text):
    return text.replace('\n', ' ').strip()

@st.cache_resource(show_spinner=False)
def get_vectorstore(chunks):
    return embed_chunks(chunks)

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
uploaded_file = st.file_uploader("📤 Upload an Arabic PDF", type=["pdf"])

if uploaded_file:
    st.success("✅ PDF uploaded successfully!")

    # # ✅ Test translation call (for debug)
    # with st.spinner("🌐 Testing translation function..."):
    #     test_text = "محتوى الوثيقة هو سرد تاريخي عن الدولة السعودية، والذكرى لملك عبد العزيز بن عبد الرحمن الفيصل، والشخصية السعودية محمد بن سعود الملقب بالمغفور له."
    #     translation_result = translate_text(test_text)
    #     st.markdown("🔁 **Test Translation Result:**")
    #     st.markdown(f"📘 Original Arabic: `{test_text}`")
    #     st.markdown(f"📗 English Translation: `{translation_result}`")

    # 🔍 Extract and clean text
    with st.spinner("🧼 Extracting and cleaning text..."):
        pdf_text = extract_text_from_pdf(uploaded_file)

    # with st.expander("📖 Preview Cleaned Text"):
    #     st.text_area("First 2000 characters of cleaned text:", value=pdf_text[:2000], height=300)

    # 🔄 Split into chunks
    with st.spinner("🔄 Splitting text into chunks..."):
        chunks = chunk_text(pdf_text)
        # st.write(f"🔹 Total Chunks Created: {len(chunks)}")

    # 🧠 Generate and store embeddings
    with st.spinner("🧠 Embedding text and storing in vector DB..."):
        vectorstore = get_vectorstore(chunks)
        # st.success("✅ Embeddings successfully stored!")

    # # 🔍 Simulated retrieval preview
    # with st.expander("🧠 Example Retrieval"):
    #     sample_query = "ما هو موضوع الوثيقة؟"
    #     st.write(f"🔍 Example Query: `{sample_query}`")
    #     docs = vectorstore.similarity_search(sample_query, k=4)
    #     for i, doc in enumerate(docs, 1):
    #         st.markdown(f"**Document {i}:**\n{doc.page_content[:500]}")

    # ---------------------------------
    # 💬 Interactive Q&A Chat Interface
    # ---------------------------------
    st.divider()
    st.subheader("💬 Ask a question based on the uploaded PDF")

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
            retrieved_docs = retrieve_similar_chunks(vectorstore, user_input, k=6)
            context = "\n\n".join(doc.page_content for doc in retrieved_docs)
            # context = "\n\n".join(clean_arabic_text(doc.page_content) for doc in retrieved_docs)

            print(context)

            prompt = (
                f"السؤال:\n{user_input}\n\n"
                f"محتوى الوثيقة:\n{context}\n\n"
                "أجب إجابة كاملة وشاملة مستندًا فقط إلى محتوى الوثيقة أعلاه."
                "يجب أن تكون إجابتك باللغة العربية الفصحى فقط دون أي كلمات إنجليزية أو لغات أخرى."
                "إذا لم يكن هناك معلومات كافية، قل ذلك بالعربية فقط دون تأليف."
            )

            response = query_llama3(prompt)
            # response = query_jais(prompt)

        response_key = f"translated_response_{len(st.session_state.messages)}"

        with st.chat_message("assistant"):
            st.markdown(response)

        # ✅ Test translation call (for debug)
        with st.spinner("Translating to English..."):
            translation_result = translate_text(response)
            st.markdown(f"📗 English Translation: `{translation_result}`")

        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("⬆️ الرجاء رفع ملف PDF لبدء المحادثة.")
