import re
import os
from textblob import TextBlob
from langdetect import detect, DetectorFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from argostranslate import translate, package
from spellchecker import SpellChecker
import logging
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
import nltk

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Set up environment
os.environ['HF_HOME'] = 'C:/Users/HP/projecthack/huggingface'

# Initialize LLM with error handling
try:
    llm = HuggingFacePipeline.from_model_id(
        model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        task="text-generation",
        model_kwargs={"temperature": 0.1, "max_length": 512}
    )
    model = ChatHuggingFace(llm=llm)
except Exception as e:
    print(f"Error initializing LLM: {e}")
    model = None

# Rest of your code remains the same...