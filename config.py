"""
============================================
Josh Talks AI/ML Speech & Audio Internship
Shared Configuration and Utility Functions
============================================

This module provides:
- Dynamic path resolution (no hardcoded paths)
- GCS URL correction logic
- Shared constants for all 4 tasks
- Helper functions for data loading and preprocessing
"""

import os
import re
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


# ============================================
# PATH CONFIGURATION (Dynamic - No Hardcoding)
# ============================================

def get_project_root() -> Path:
    """
    Get the project root directory dynamically.
    Returns the directory containing this config.py file.
    """
    return Path(__file__).parent.resolve()


def get_data_path(filename: str) -> Path:
    """
    Get the full path to a data file in the project root.
    
    Args:
        filename: Name of the file (e.g., 'FT Data.xlsx')
    
    Returns:
        Full Path object to the file
    """
    return get_project_root() / filename


def get_output_path(filename: str, create_dir: bool = True) -> Path:
    """
    Get the full path to an output file in the outputs directory.
    
    Args:
        filename: Name of the output file
        create_dir: Whether to create the outputs directory if it doesn't exist
    
    Returns:
        Full Path object to the output file
    """
    output_dir = get_project_root() / "outputs"
    if create_dir:
        output_dir.mkdir(exist_ok=True)
    return output_dir / filename


def get_model_path(model_name: str, create_dir: bool = True) -> Path:
    """
    Get the full path to a model directory.
    
    Args:
        model_name: Name of the model directory
        create_dir: Whether to create the directory if it doesn't exist
    
    Returns:
        Full Path object to the model directory
    """
    model_dir = get_project_root() / "models" / model_name
    if create_dir:
        model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


# ============================================
# GCS URL CORRECTION (Task 1, 2, 3, 4)
# ============================================

# URL mapping constants
GCS_OLD_BASE = "storage.googleapis.com/joshtalks-data-collection/hq_data/hi/"
GCS_NEW_BASE = "storage.googleapis.com/upload_goai/"

GCS_TESTING_OLD = "storage.googleapis.com/testing_audio_for_josh/"
GCS_TESTING_NEW = "storage.googleapis.com/testing_audio_for_josh/"  # No change for testing audio


def correct_gcs_url(url: str) -> str:
    """
    Correct non-functional GCS URLs as per task instructions.
    
    Changes:
    - joshtalks-data-collection/hq_data/hi/{id1}/{id2}_file.ext
    - TO: upload_goai/{id1}/{id2}_file.ext
    
    Args:
        url: Original GCS URL (may be incorrect)
    
    Returns:
        Corrected GCS URL
    """
    if pd.isna(url) or not url:
        return url
    
    url = str(url).strip()
    
    # Apply correction for main dataset URLs
    if GCS_OLD_BASE in url:
        # Extract the ID parts from the old URL pattern
        # Pattern: .../hq_data/hi/{id1}/{id2}_file.ext
        match = re.search(r'hq_data/hi/(\d+)/(\d+)_(\w+\.\w+)', url)
        if match:
            id1, id2, filename = match.groups()
            return f"https://storage.googleapis.com/upload_goai/{id1}/{id2}_{filename}"
    
    # Testing audio URLs (Task 4) - no correction needed
    if GCS_TESTING_OLD in url:
        return url
    
    return url


def correct_gcs_urls_in_dataframe(df: pd.DataFrame, 
                                   url_columns: List[str] = None) -> pd.DataFrame:
    """
    Correct GCS URLs in all specified columns of a DataFrame.
    
    Args:
        df: Input DataFrame containing URL columns
        url_columns: List of column names containing URLs. 
                     If None, auto-detects common URL column names.
    
    Returns:
        DataFrame with corrected URLs
    """
    df = df.copy()
    
    # Auto-detect URL columns if not specified
    if url_columns is None:
        url_columns = [col for col in df.columns if 'url' in col.lower()]
    
    for col in url_columns:
        if col in df.columns:
            df[col] = df[col].apply(correct_gcs_url)
    
    return df


# ============================================
# TASK-SPECIFIC CONFIGURATIONS
# ============================================

# Task 1: Fine-tuning Configuration
TASK1_CONFIG = {
    'model_name': 'openai/whisper-small',
    'language': 'hindi',
    'task': 'transcribe',
    'sampling_rate': 16000,
    'max_duration': 30.0,  # seconds
    'batch_size': 8,
    'learning_rate': 1e-5,
    'num_epochs': 10,
    'warmup_steps': 500,
    'evaluation_strategy': 'epoch',
    'save_strategy': 'epoch',
    'wer_metric': 'jiwer',
    'test_dataset': 'google/fleurs',
    'test_split': 'test',
    'test_language': 'hi_in',
}

# Task 2: Cleanup Pipeline Configuration
TASK2_CONFIG = {
    'number_normalization': {
        'enabled': True,
        'precision_over_recall': True,
        'idiom_blacklist': ['दो-चार', 'छै सात', 'एक आधे', 'छः सात'],
    },
    'english_detection': {
        'enabled': True,
        'tag_format': '[EN]{word}[/EN]',
        'loanword_lexicon_path': 'loanwords.json',
    },
}

# Task 3: Spelling Validation Configuration
TASK3_CONFIG = {
    'total_unique_words': 177509,
    'confidence_levels': ['high', 'medium', 'low'],
    'review_sample_size': 50,
    'dictionary_sources': ['hindi_wordnet', 'indic_nlp'],
    'frequency_thresholds': {
        'high_freq': 1000,
        'low_freq': 5,
    },
}

# Task 4: Lattice Evaluation Configuration
TASK4_CONFIG = {
    'num_models': 6,  # Human + 5 ASR models
    'model_names': ['Human', 'H', 'i', 'k', 'l', 'm', 'n'],
    'consensus_threshold': 4,  # Minimum models needed to override reference
    'alignment_unit': 'word',
    'wer_calculation': 'best_path',
}

# ============================================
# HINDI NUMBER MAPPING (Task 2)
# ============================================

HINDI_NUMBERS = {
    # Single digits
    'एक': '1', 'दो': '2', 'तीन': '3', 'चार': '4', 'पाँच': '5', 'पांच': '5',
    'छै': '6', 'छह': '6', 'सात': '7', 'आठ': '8', 'नौ': '9', 'दस': '10',
    
    # Tens
    'बीस': '20', 'तीस': '30', 'चालीस': '40', 'पचास': '50',
    'साठ': '60', 'सत्तर': '70', 'अस्सी': '80', 'नब्बे': '90',
    
    # Hundreds and above
    'सौ': '100', 'हज़ार': '1000', 'हजार': '1000',
    'लाख': '100000', 'करोड़': '10000000',
    
    # Irregular numbers
    'पच्चीस': '25', 'अड़तीस': '38', 'चौवन': '54', 'चौदह': '14',
    'बारह': '12', 'तेरह': '13', 'पंद्रह': '15', 'सोलह': '16',
    'सत्रह': '17', 'अठारह': '18', 'उन्नीस': '19',
}

# ============================================
# ENGLISH LOANWORD LEXICON (Task 2, 3)
# ============================================

COMMON_ENGLISH_LOANWORDS = {
    'प्रोजेक्ट', 'एरिया', 'टेंट', 'लाइट', 'मिस्टेक', 'कैम्प', 'गार्ड',
    'रोड', 'जंगल', 'फोन', 'कंप्यटूर', 'इंटरव्यू', 'जॉब', 'प्रॉब्लम',
    'सॉफ्टवेयर', 'हार्डवेयर', 'इंटरनेट', 'मोबाइल', 'लैपटॉप', 'डेस्कटॉप',
    'फीडबैक', 'ट्रेडिशनल', 'डिश', 'डिशेस', 'अनहेल्दी', 'हेल्दी',
    'प्योर', 'हार्ट', 'इनफॉर्मेशन', 'पार्किंग', 'फ्लोर', 'लिफ्ट',
    'स्ट्रगल', 'पसंदीदा', 'खिचड़ी', 'इंफॉर्मेशन', 'गिफ्टेड', 'लैंड',
    'एक्सप्लोर', 'म्यूजिक', 'डांसिंग', 'पैशन', 'एक्चुअली',
}

# ============================================
# DATA LOADING HELPERS
# ============================================

def load_ft_data() -> pd.DataFrame:
    """
    Load Task 1 FT Data.xlsx with URL correction applied.
    
    Returns:
        DataFrame with corrected GCS URLs
    """
    data_path = get_data_path('FT Data.xlsx')
    df = pd.read_excel(data_path)
    df = correct_gcs_urls_in_dataframe(df)
    return df


def load_unique_words() -> pd.DataFrame:
    """
    Load Task 3 Unique Words Data.xlsx.
    
    Returns:
        DataFrame with unique words
    """
    data_path = get_data_path('Unique Words Data.xlsx')
    df = pd.read_excel(data_path)
    return df


def load_question4_data() -> pd.DataFrame:
    """
    Load Task 4 Question 4.xlsx with segment URLs.
    
    Returns:
        DataFrame with segment data
    """
    data_path = get_data_path('Question 4.xlsx')
    df = pd.read_excel(data_path)
    return df


def load_transcription_json(url: str) -> Optional[List[Dict[str, Any]]]:
    """
    Load and parse a transcription JSON file from GCS URL.
    
    Args:
        url: GCS URL to the transcription JSON file
    
    Returns:
        List of transcription segments, or None if loading fails
    """
    import requests
    
    try:
        corrected_url = correct_gcs_url(url)
        response = requests.get(corrected_url, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error loading transcription from {url}: {e}")
        return None


def concatenate_transcription_segments(segments: List[Dict[str, Any]]) -> str:
    """
    Concatenate all text segments from a transcription JSON into a single string.
    
    Args:
        segments: List of segment dictionaries with 'text' key
    
    Returns:
        Concatenated transcription text
    """
    if not segments:
        return ""
    
    texts = [seg.get('text', '').strip() for seg in segments if seg.get('text')]
    return " ".join(texts)


# ============================================
# TEXT NORMALIZATION HELPERS
# ============================================

def normalize_hindi_text(text: str) -> str:
    """
    Normalize Hindi text for consistent comparison.
    
    Args:
        text: Input Hindi text
    
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove punctuation for comparison (optional)
    # text = re.sub(r'[^\w\s]', '', text)
    
    return text


def tokenize_hindi(text: str) -> List[str]:
    """
    Tokenize Hindi text into words.
    
    Args:
        text: Input Hindi text
    
    Returns:
        List of word tokens
    """
    if not text:
        return []
    
    # Simple whitespace tokenization
    # For production, use indic-nlp-lib for better Hindi tokenization
    return text.strip().split()


# ============================================
# EVALUATION METRICS
# ============================================

def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate between reference and hypothesis.
    
    Args:
        reference: Ground truth text
        hypothesis: Model output text
    
    Returns:
        WER as a float (e.g., 0.30 = 30%)
    """
    try:
        from jiwer import wer
        return wer(reference, hypothesis)
    except ImportError:
        # Fallback simple implementation
        ref_words = tokenize_hindi(normalize_hindi_text(reference))
        hyp_words = tokenize_hindi(normalize_hindi_text(hypothesis))
        
        # Simple edit distance calculation
        edits = abs(len(ref_words) - len(hyp_words))
        return edits / max(len(ref_words), 1)


# ============================================
# LOGGING AND UTILITIES
# ============================================

def setup_logging(log_file: str = None) -> None:
    """
    Setup logging for the project.
    
    Args:
        log_file: Optional log file path. If None, logs to console only.
    """
    import logging
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(get_output_path(log_file)),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[logging.StreamHandler()]
        )


def save_results_to_excel(df: pd.DataFrame, filename: str) -> Path:
    """
    Save results DataFrame to Excel file in outputs directory.
    
    Args:
        df: DataFrame to save
        filename: Output filename
    
    Returns:
        Path to saved file
    """
    output_path = get_output_path(filename)
    df.to_excel(output_path, index=False)
    return output_path


def save_results_to_csv(df: pd.DataFrame, filename: str) -> Path:
    """
    Save results DataFrame to CSV file in outputs directory.
    
    Args:
        df: DataFrame to save
        filename: Output filename
    
    Returns:
        Path to saved file
    """
    output_path = get_output_path(filename)
    df.to_csv(output_path, index=False)
    return output_path


# ============================================
# VALIDATION FUNCTIONS
# ============================================

def validate_devanagari_structure(word: str) -> bool:
    """
    Validate if a word has valid Devanagari character structure.
    
    Args:
        word: Hindi word in Devanagari script
    
    Returns:
        True if structure is valid, False otherwise
    """
    if not word:
        return False
    
    # Check for invalid character combinations
    # This is a simplified check - production would use indic-nlp-lib
    
    # Check for repeated vowels (often indicates error)
    if re.search(r'[ाीीूूेेैैोोौौ]{2,}', word):
        return False
    
    # Check for invalid consonant clusters
    # (Simplified - would need proper linguistic rules)
    
    return True


def is_english_loanword(word: str) -> bool:
    """
    Check if a word is a common English loanword in Hindi.
    
    Args:
        word: Hindi word (potentially transliterated English)
    
    Returns:
        True if word is in loanword lexicon
    """
    return word in COMMON_ENGLISH_LOANWORDS


# ============================================
# GRADIO CONFIGURATION
# ============================================

GRADIO_CONFIG = {
    'server_name': '0.0.0.0',
    'server_port': 7860,
    'share': False,
    'debug': True,
}


def get_gradio_config() -> Dict[str, Any]:
    """
    Get Gradio launch configuration.
    
    Returns:
        Dictionary of Gradio configuration parameters
    """
    return GRADIO_CONFIG.copy()


# ============================================
# MAIN EXECUTION (Testing)
# ============================================

if __name__ == "__main__":
    """
    Test the config module functionality.
    """
    print("=" * 60)
    print("Josh Talks AI/ML Internship - Config Module Test")
    print("=" * 60)
    
    # Test path resolution
    print(f"\nProject Root: {get_project_root()}")
    print(f"Data Path Example: {get_data_path('FT Data.xlsx')}")
    print(f"Output Path Example: {get_output_path('test_output.csv')}")
    
    # Test URL correction
    test_url = "https://storage.googleapis.com/joshtalks-data-collection/hq_data/hi/967179/825780_audio.wav"
    corrected = correct_gcs_url(test_url)
    print(f"\nOriginal URL: {test_url}")
    print(f"Corrected URL: {corrected}")
    
    # Test Hindi number mapping
    print(f"\nHindi Numbers Mapping Sample: {list(HINDI_NUMBERS.items())[:5]}")
    
    # Test text normalization
    test_text = "नमस्ते   दुनिया"
    normalized = normalize_hindi_text(test_text)
    print(f"\nOriginal Text: '{test_text}'")
    print(f"Normalized Text: '{normalized}'")
    
    print("\n" + "=" * 60)
    print("Config Module Test Complete!")
    print("=" * 60)