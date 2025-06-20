import re
import os
from textblob import TextBlob
from langdetect import detect, DetectorFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from argostranslate import translate, package
from spellchecker import SpellChecker
import logging

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load a language detection model (based on XLM-RoBERTa)
model_name = "papluca/xlm-roberta-base-language-detection"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def detect_language(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1)
    label = model.config.id2label[pred.item()]
    print(f"Detected Language: {label}")
    return label





logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# For consistent langdetect results
DetectorFactory.seed = 0

# Initialize spell checker
spell = SpellChecker()

# Pre-compile regex patterns for better performance
URL_PATTERN = re.compile(r"http\S+")
NON_ALPHA_PATTERN = re.compile(r"[^a-zA-Z\s]")

# Categorization keywords
CATEGORY_KEYWORDS = {
    "delivery": ["late", "delay", "slow", "delivered", "shipping"],
    "product": ["quality", "defect", "broken", "damage"],
    "price": ["expensive", "cheap", "cost", "worth", "price"],
}

def clean_text(text):
    """More efficient text cleaning"""
    if not text:
        return ""
        
    text = URL_PATTERN.sub("", text)
    text = NON_ALPHA_PATTERN.sub("", text)
    return text.lower().strip()

def correct_spelling(text):
    """More robust spelling correction with error handling"""
    if not text:
        return ""
        
    try:
        words = word_tokenize(text)
        corrected = [spell.correction(word) or word for word in words]
        return " ".join(filter(None, corrected))  # Filter out None values
    except Exception as e:
        logger.warning(f"Spell correction failed: {str(e)}")
        return text

def analyze_sentiment(text):
    """Enhanced sentiment analysis with error handling"""
    if not text:
        return "neutral"
        
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.2:  # Slightly higher threshold for more confidence
            return "positive"
        elif polarity < -0.2:
            return "negative"
        return "neutral"
    except Exception as e:
        logger.warning(f"Sentiment analysis failed: {str(e)}")
        return "neutral"

def categorize_feedback(text):
    """Improved categorization with word boundaries"""
    if not text:
        return "general"
        
    text = text.lower()
    words = set(word_tokenize(text))
    
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in words for keyword in keywords):
            return category
    return "general"

def translate_to_english(text, source_lang="auto", target_lang="en"):
    """More robust translation with better error handling"""
    if not text or source_lang == target_lang:
        return text
        
    try:
        installed_languages = translate.get_installed_languages()
        from_lang = next((x for x in installed_languages if x.code == source_lang), None)
        to_lang = next((x for x in installed_languages if x.code == target_lang), None)

        if from_lang and to_lang:
            translation = from_lang.get_translation(to_lang)
            if translation:
                return translation.translate(text)
    except Exception as e:
        logger.error(f"Translation failed from {source_lang} to {target_lang}: {str(e)}")
    
    return text

def process_feedback(text):
    """Main processing function with better error handling and logging"""
    if not text or not isinstance(text, str):
        return {
            "text": text,
            "cleaned": "",
            "language": "unknown",
            "sentiment": "neutral",
            "category": "general"
        }
    
    try:
        cleaned = clean_text(text)
        corrected = correct_spelling(cleaned)
        
        # Detect language before translation
        detected_lang = detect_language(corrected)
        
        # Only translate if not English
        if detected_lang != "en":
            corrected = translate_to_english(corrected, source_lang=detected_lang)
            # Re-detect language after translation
            detected_lang = detect_language(corrected)
        
        sentiment = analyze_sentiment(corrected)
        category = categorize_feedback(corrected)
        
        return {
            "text": text,
            "cleaned": corrected,
            "language": detected_lang,
            "sentiment": sentiment,
            "category": category
        }
        
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        return {
            "text": text,
            "cleaned": "",
            "language": "unknown",
            "sentiment": "neutral",
            "category": "general"
        }

def process_all(feedback_list):
    """Batch processing with progress tracking"""
    if not feedback_list:
        return []
        
    return [process_feedback(text) for text in feedback_list]