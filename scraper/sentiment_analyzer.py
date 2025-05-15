from transformers import pipeline
from typing import Dict
import logging

class SentimentAnalyzer:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize sentiment analysis pipeline
        :param model_name: HuggingFace model to use (default is a lightweight sentiment model)
        """
        try:
            self.nlp = pipeline("sentiment-analysis", model=model_name)
            logging.info("Sentiment analyzer initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize sentiment analyzer: {str(e)}")
            raise

    def analyze_text(self, text: str) -> Dict:
        """
        Analyze sentiment for a single text
        :param text: Text to analyze
        :return: Dictionary with 'label' and 'score' keys
        """
        if not text:
            return {'label': 'NEUTRAL', 'score': 0.5}
        
        try:
            # Process text (limit to first 512 characters due to model constraints)
            processed_text = text[:512]
            result = self.nlp(processed_text)[0]  # Get first result
            return {
                'label': result['label'],
                'score': result['score']
            }
        except Exception as e:
            logging.error(f"Sentiment analysis failed: {str(e)}")
            return {'label': 'NEUTRAL', 'score': 0.5}