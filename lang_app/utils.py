import requests
import urllib.parse
import time
from googletrans import Translator

class GoogleTranslator:
    """
    Wrapper class for Google Translator.
    """
    def __init__(self):
        self.translator = Translator()

    def translate(self, text, target_lang='en', source_lang='auto'):
        """
        Translate text using Google Translator.
        """
        try:
            result = self.translator.translate(text, src=source_lang, dest=target_lang)
            return result.text
        except Exception as e:
            raise RuntimeError(f"Translation failed: {e}")

def translate(text, source_lang="en", target_lang="ko", delay=1):
    """
    Translate text using the Lingva API.
    """
    encoded_text = urllib.parse.quote(text)
    url = f"https://lingva.ml/api/v1/{source_lang}/{target_lang}/{encoded_text}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        time.sleep(delay)
        return response.json()["translation"]
    except requests.exceptions.RequestException as e:
        print(f"Translation API error: {e}")
        raise RuntimeError(f"Lingva API failed: {e}")

def translate_ko(text, delay=1):
    """
    Shortcut for translating text from English to Korean.
    """
    return translate(text, source_lang="en", target_lang="ko", delay=delay)

def translate_with_fallback(text, target_language='ko', source_language='en'):
    """
    Translate text using Lingva API with Google Translate fallback.
    """
    try:
        return translate(text, source_lang=source_language, target_lang=target_language)
    except Exception as e:
        print(f"Lingva API translation failed: {e}")
        try:
            google_translator = GoogleTranslator()
            return google_translator.translate(text, target_lang=target_language, source_lang=source_language)
        except Exception as google_e:
            print(f"Google Translate failed: {google_e}")
            return text  # Return original text as a last resort
