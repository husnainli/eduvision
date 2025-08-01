import re

def clean_arabic_text(text):
    """Cleans and normalizes Arabic text for processing."""
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)  # Remove non-Arabic chars
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()
