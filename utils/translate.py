# utils/translate.py
from deep_translator import GoogleTranslator

def translate_text(text, source_lang="ar", target_lang="en"):
    """
    Translates text from source_lang to target_lang using Google Translate via deep-translator.
    """
    try:
        translated = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
        return translated
    except Exception as e:
        return f"❌ Translation failed: {e}"
