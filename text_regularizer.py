from typing import List, Set
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class TextRegularizer:
    def __init__(self):
        # Load spaCy model for NLP tasks
        self.nlp = spacy.load("en_core_web_sm")
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        # Thresholds for filtering
        self.similarity_threshold = 0.7
        self.min_words = 4
        self.max_words = 25
        
    def _get_key_phrases(self, text: str) -> Set[str]:
        """Extract key noun phrases and entities from text"""
        doc = self.nlp(text)
        phrases = set()
        
        # Get noun phrases
        for chunk in doc.noun_chunks:
            phrases.add(chunk.text.lower())
            
        # Get named entities
        for ent in doc.ents:
            phrases.add(ent.text.lower())
            
        return phrases

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts"""
        try:
            tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
            return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            return 1.0 if text1.lower() == text2.lower() else 0.0

    def regularize_assumptions(self, original_text: str, assumptions: List[str]) -> List[str]:
        """Filter and regularize assumptions based on original text"""
        if not assumptions:
            return []
            
        key_phrases = self._get_key_phrases(original_text)
        regularized = []
        
        for assumption in assumptions:
            # Basic length check
            words = assumption.split()
            if len(words) < self.min_words or len(words) > self.max_words:
                continue
                
            # Check similarity with original text
            similarity = self._compute_similarity(original_text, assumption)
            if similarity > self.similarity_threshold:
                continue
                
            # Check if assumption contains at least one key phrase
            assumption_phrases = self._get_key_phrases(assumption)
            if not assumption_phrases.intersection(key_phrases):
                continue
                
            regularized.append(assumption)
            
        return regularized
