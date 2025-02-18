from typing import List, Tuple, Dict
from difflib import SequenceMatcher
import re

def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Convert to lowercase and remove extra whitespace
    text = text.lower().strip()
    # Remove punctuation except hyphens
    text = re.sub(r'[^\w\s-]', '', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    return text

def calculate_similarity(str1: str, str2: str) -> float:
    """Calculate string similarity using SequenceMatcher."""
    str1 = normalize_text(str1)
    str2 = normalize_text(str2)
    return SequenceMatcher(None, str1, str2).ratio()

def find_best_match(query: str, candidates: List[str], threshold: float = 0.8) -> Tuple[str, float]:
    """Find the best matching string from a list of candidates."""
    best_match = None
    best_score = 0.0
    
    query = normalize_text(query)
    
    for candidate in candidates:
        score = calculate_similarity(query, candidate)
        if score > best_score:
            best_score = score
            best_match = candidate
    
    if best_score >= threshold:
        return best_match, best_score
    return None, best_score

def find_matching_modifications(text: str, available_mods: List[str], threshold: float = 0.8) -> List[str]:
    """Find matching modifications in text."""
    text = normalize_text(text)
    matches = []
    
    for mod in available_mods:
        if calculate_similarity(text, mod) >= threshold:
            matches.append(mod)
    
    return matches

def extract_quantities(text: str) -> Dict[str, int]:
    """Extract quantities from text using regex patterns."""
    # Common number words and their values
    number_words = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }
    
    # Convert number words to digits
    for word, num in number_words.items():
        text = re.sub(rf'\b{word}\b', str(num), text.lower())
    
    # Find quantities with items
    quantities = {}
    patterns = [
        r'(\d+)\s*(?:x\s*)?([a-zA-Z\s]+)',  # "2 club sandwich" or "2x club sandwich"
        r'([a-zA-Z\s]+?)(?:\s*x\s*)?(\d+)',  # "club sandwich 2" or "club sandwich x2"
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            num, item = match.groups() if pattern.startswith(r'(\d+)') else match.groups()[::-1]
            item = normalize_text(item)
            quantities[item] = int(num)
    
    return quantities 
