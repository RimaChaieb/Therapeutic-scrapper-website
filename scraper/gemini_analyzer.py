import os
import requests
import logging
import json
from typing import Optional

class GeminiAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        
        if not self.api_key:
            self.logger.warning("Gemini API key not configured. Set GEMINI_API_KEY environment variable.")

    def generate_insight(self, text: str) -> str:
        """Generate therapeutic insights using Gemini API"""
        if not self.api_key:
            return "Analysis unavailable - API key not configured"

        # Sanitize input text - ensure it's not empty
        if not text or text.strip() == '':
            return "Analysis unavailable - Empty content"

        headers = {'Content-Type': 'application/json'}
        prompt = (
            "Analyze this Reddit post from a mental health professional perspective. "
            "Identify key emotional themes, potential concerns, and provide "
            "supportive, clinically-informed insights. Keep response concise (3-4 sentences).\n\n"
            f"Post content: {text[:2000]}"  # Limit to first 2000 chars
        )
        
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                },
                 {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_CIVIC_INTEGRITY",
                    "threshold": "BLOCK_NONE"
                }
            ],
            "generationConfig": {
                "temperature": 0.5,
                "maxOutputTokens": 256
            }
        }
        
        try:
            self.logger.debug(f"Sending request to Gemini API with {len(text)} characters of text")
            self.logger.debug(f"Request data: {json.dumps(data, indent=2)}")  # Log the data

            response = requests.post(
                f"{self.base_url}?key={self.api_key}",
                headers=headers,
                json=data,
                timeout=15
            )
            
            self.logger.debug(f"Gemini API response status: {response.status_code}")
            
            if response.status_code != 200:
                self.logger.error(f"Gemini API HTTP error: {response.status_code} - {response.text}")
                return f"Analysis unavailable - API returned status {response.status_code}"
            
            try:
                result = response.json()
            except json.JSONDecodeError:
                self.logger.error(f"Gemini API returned invalid JSON: {response.text}")
                return "Analysis unavailable - Invalid response format"
            
            # Debug the response structure if needed
            # self.logger.debug(f"Gemini API response: {json.dumps(result, indent=2)}")
            
            candidates = result.get('candidates', [])
            if not candidates:
                self.logger.error("Gemini API returned no candidates")
                return "Analysis unavailable - No response from API"
                
            content = candidates[0].get('content', {})
            parts = content.get('parts', [])
            
            if not parts:
                self.logger.error("Gemini API response missing parts")
                return "Analysis unavailable - Malformed API response"
                
            insight_text = parts[0].get('text', '')
            
            if not insight_text:
                self.logger.error("Gemini API returned empty text")
                return "Analysis unavailable - Empty response from API"
                
            return insight_text
            
        except requests.exceptions.Timeout:
            self.logger.error("Gemini API request timed out")
            return "Analysis unavailable - API request timed out"
            
        except requests.exceptions.ConnectionError:
            self.logger.error("Gemini API connection error")
            return "Analysis unavailable - Could not connect to API"
            
        
        except Exception as e:
            self.logger.error(f"Gemini API error: {str(e)}")
            return f"Analysis unavailable - {str(e)}"