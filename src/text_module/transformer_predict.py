from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import os

# Use pipelines for simplicity. Models will be downloaded on first run.
sentiment_classifier = pipeline('sentiment-analysis')
summarizer = pipeline('summarization')

def predict_sentiment(text):
    try:
        res = sentiment_classifier(text[:512])
        return res[0]['label']
    except Exception as e:
        return f'error: {e}'

def summarize_text(text):
    try:
        out = summarizer(text, max_length=80, min_length=20, do_sample=False)
        return out[0]['summary_text']
    except Exception as e:
        return f'error: {e}'
