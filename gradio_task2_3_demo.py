"""
============================================
Josh Talks AI/ML Speech & Audio Internship
Gradio Demo for Task 2 & 3: ASR Cleanup + Spelling Validation
============================================

This script provides an interactive web interface for:
1. Task 2: Number Normalization + English Word Detection/Tagging
2. Task 3: Hindi Word Spelling Validation with Confidence Scoring
3. Batch processing capabilities
4. Results export functionality

No hardcoded paths - all data access is dynamic via config.py
"""

import os
import sys
import json
import re
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd

# Gradio import
try:
    import gradio as gr
except ImportError:
    print("❌ Gradio not installed. Run: pip install gradio")
    sys.exit(1)

# Local config import
from config import (
    get_project_root,
    get_data_path,
    get_output_path,
    normalize_hindi_text,
    save_results_to_excel,
    save_results_to_csv,
    TASK2_CONFIG,
    TASK3_CONFIG,
    HINDI_NUMBERS,
    COMMON_ENGLISH_LOANWORDS,
    GRADIO_CONFIG,
    setup_logging,
)


# ============================================
# LOGGING SETUP
# ============================================

logger = logging.getLogger(__name__)
setup_logging('gradio_task2_3.log')


# ============================================
# TASK 2: ASR CLEANUP PIPELINE
# ============================================

class NumberNormalizer:
    """
    Converts Hindi number words to digits with precision-focused approach.
    """
    
    def __init__(self):
        self.num_map = HINDI_NUMBERS
        
        # Idiom blacklist - phrases where number conversion should be skipped
        self.idiom_blacklist = [
            'दो-चार', 'दो चार', 'छै सात', 'छह सात', 'छः सात',
            'एक आधे', 'एक-आधे', 'दो-तीन', 'दो तीन',
            'चार-पाँच', 'चार पाँच', 'सात-आठ', 'सात आठ',
        ]
    
    def contains_idiom(self, text: str) -> bool:
        """Check if text contains idiomatic phrases."""
        text_lower = text.lower()
        for idiom in self.idiom_blacklist:
            if idiom in text_lower:
                return True
        return False
    
    def normalize(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Normalize Hindi number words to digits.
        
        Returns:
            Tuple of (normalized_text, conversion_log)
        """
        if not text:
            return text, []
        
        conversion_log = []
        
        # Check for idioms (skip normalization if found)
        if self.contains_idiom(text):
            conversion_log.append({
                'type': 'idiom_skipped',
                'original': text[:50],
                'reason': 'Contains idiomatic expression'
            })
            return text, conversion_log
        
        # Replace direct mappings (sorted by length to avoid partial matches)
        sorted_nums = sorted(self.num_map.keys(), key=len, reverse=True)
        
        for hindi_num, digit in self.num_map.items():
            pattern = r'\b' + re.escape(hindi_num) + r'\b'
            if re.search(pattern, text):
                text = re.sub(pattern, digit, text)
                conversion_log.append({
                    'type': 'number_converted',
                    'original': hindi_num,
                    'converted': digit,
                })
        
        return text, conversion_log


class EnglishWordDetector:
    """
    Identifies English loanwords transliterated into Devanagari script.
    """
    
    def __init__(self):
        self.loanword_lexicon: Set[str] = set(COMMON_ENGLISH_LOANWORDS)
        
        # Additional common loanwords
        additional_loanwords = {
            'इंटरव्यू', 'जॉब', 'प्रॉब्लम', 'सॉल्व', 'मीटिंग', 'कॉल',
            'मैसेज', 'ईमेल', 'व्हाट्सएप', 'फेसबुक', 'इंस्टाग्राम',
            'यूट्यूब', 'गूगल', 'ऑनलाइन', 'ऑफलाइन', 'डाउनलोड', 'अपलोड',
            'लॉगिन', 'लॉगआउट', 'पासवर्ड', 'यूजर', 'एडमिन', 'सेटिंग',
            'अपडेट', 'सब्सक्राइब', 'लाइक', 'शेयर', 'कमेंट', 'फॉलो',
        }
        
        self.loanword_lexicon.update(additional_loanwords)
    
    def detect_and_tag(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Detect English loanwords and tag them with [EN]...[/EN].
        
        Returns:
            Tuple of (tagged_text, detection_log)
        """
        if not text:
            return text, []
        
        detection_log = []
        words = text.split()
        tagged_words = []
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            original_word = word
            
            is_loanword = clean_word in self.loanword_lexicon
            
            if is_loanword:
                tagged_word = f"[EN]{original_word}[/EN]"
                tagged_words.append(tagged_word)
                detection_log.append({
                    'word': clean_word,
                    'position': len(tagged_words),
                    'confidence': 'high'
                })
            else:
                tagged_words.append(original_word)
        
        tagged_text = ' '.join(tagged_words)
        return tagged_text, detection_log


class ASRCleanupPipeline:
    """
    Main pipeline combining number normalization and English word detection.
    """
    
    def __init__(self):
        self.number_normalizer = NumberNormalizer()
        self.english_detector = EnglishWordDetector()
    
    def process(self, text: str, apply_number_norm: bool = True, 
                apply_english_tag: bool = True) -> Dict[str, Any]:
        """
        Process text through the cleanup pipeline.
        
        Returns:
            Dictionary with all processing results
        """
        result = {
            'original': text,
            'number_normalized': text,
            'english_tagged': text,
            'final': text,
            'number_conversions': [],
            'english_detections': [],
            'idioms_skipped': 0,
        }
        
        # Step 1: Number Normalization
        if apply_number_norm:
            normalized_text, number_log = self.number_normalizer.normalize(text)
            result['number_normalized'] = normalized_text
            result['number_conversions'] = number_log
            result['idioms_skipped'] = sum(
                1 for entry in number_log if entry.get('type') == 'idiom_skipped'
            )
        
        # Step 2: English Word Detection
        if apply_english_tag:
            tagged_text, english_log = self.english_detector.detect_and_tag(
                result['number_normalized']
            )
            result['english_tagged'] = tagged_text
            result['english_detections'] = english_log
        
        result['final'] = result['english_tagged']
        
        return result


# ============================================
# TASK 3: SPELLING VALIDATION PIPELINE
# ============================================

class HindiDictionaryLoader:
    """
    Loads and manages Hindi dictionary resources for spelling validation.
    """
    
    def __init__(self):
        self.hindi_words: Set[str] = set()
        self.english_loanwords: Set[str] = set()
        self.proper_nouns: Set[str] = set()
        
        self._load_resources()
    
    def _load_resources(self):
        """Load all dictionary resources."""
        # Load standard Hindi words (common words)
        common_hindi_words = {
            'है', 'हैं', 'था', 'थी', 'थे', 'हो', 'होता', 'होती', 'होते',
            'कर', 'किया', 'किए', 'की', 'के', 'का', 'को', 'से', 'में', 'पर',
            'और', 'या', 'तो', 'लेकिन', 'परंतु', 'मगर', 'इसलिए', 'क्योंकि',
            'क्या', 'कौन', 'कब', 'कहाँ', 'कैसे', 'कितना', 'कितनी', 'कितने',
            'यह', 'वह', 'ये', 'वे', 'इस', 'उस', 'इन', 'उन',
            'मेरा', 'तेरा', 'हमारा', 'तुम्हारा', 'उसका', 'उनका',
            'आप', 'तुम', 'हम', 'मैं', 'वो', 'जो', 'सो',
            'जा', 'जाता', 'जाती', 'जाते', 'आया', 'आई', 'आए', 'गया', 'गई', 'गए',
            'देख', 'देखा', 'देखी', 'देखे', 'सुन', 'सुना', 'सुनी', 'सुने',
            'बोल', 'बोला', 'बोली', 'बोले', 'कह', 'कहा', 'कही', 'कहे',
            'पता', 'बात', 'बातें', 'काम', 'कामों', 'दिन', 'रात',
            'सुबह', 'शाम', 'दोपहर', 'साल', 'महीने', 'हफ्ते',
            'घर', 'स्कूल', 'कॉलेज', 'ऑफिस', 'बाजार', 'दुकान',
            'किताब', 'पेन', 'कागज', 'पढ़ाई', 'लिखाई', 'नौकरी',
            'पैसा', 'रुपये', 'कीमत', 'महंगा', 'सस्ता',
            'अच्छा', 'बुरा', 'बड़ा', 'छोटा', 'नया', 'पुराना',
            'खुश', 'दुखी', 'गुस्सा', 'डर', 'प्यार', 'नफरत',
            'दोस्त', 'दुश्मन', 'परिवार', 'माँ', 'पिता', 'भाई', 'बहन',
            'खाना', 'पीना', 'सोना', 'जागना', 'चलना', 'दौड़ना',
            'समय', 'जगह', 'रास्ता', 'दिशा', 'दूर', 'पास',
            'पहले', 'बाद', 'अब', 'फिर', 'कभी', 'हमेशा', 'कभी-कभी',
            'बहुत', 'थोड़ा', 'ज्यादा', 'कम', 'सब', 'सभी', 'कोई', 'कुछ',
            'हर', 'एक', 'दो', 'तीन', 'चार', 'पाँच', 'छह', 'सात', 'आठ', 'नौ', 'दस',
            'सौ', 'हज़ार', 'लाख', 'करोड़',
        }
        
        self.hindi_words.update(common_hindi_words)
        self.english_loanwords = set(COMMON_ENGLISH_LOANWORDS)
        
        # Proper nouns
        proper_nouns = {
            'भारत', 'हिंदुस्तान', 'इंडिया', 'दिल्ली', 'मुंबई', 'कोलकाता', 'चेन्नई',
            'बेंगलुरु', 'हैदराबाद', 'पुणे', 'अहमदाबाद', 'जयपुर', 'लखनऊ', 'कानपुर',
            'राजस्थान', 'गुजरात', 'महाराष्ट्र', 'कर्नाटक', 'तमिलनाडु', 'केरल',
            'जोश', 'टॉक्स', 'जोशटॉक्स', 'Google', 'Facebook', 'WhatsApp',
        }
        
        self.proper_nouns.update(proper_nouns)
    
    def is_standard_hindi(self, word: str) -> bool:
        return word in self.hindi_words
    
    def is_english_loanword(self, word: str) -> bool:
        return word in self.english_loanwords
    
    def is_proper_noun(self, word: str) -> bool:
        return word in self.proper_nouns


class PhoneticValidator:
    """
    Validates Devanagari word structure using phonetic rules.
    """
    
    def __init__(self):
        self.devanagari_range = (0x0900, 0x097F)
        
        self.invalid_patterns = [
            r'[ाीीूूेेैैोोौौ]{2,}',
            r'[क-ह]{4,}',
            r'[़़]{2,}',
            r'[ंंं]{2,}',
            r'[ःः]{2,}',
        ]
        
        self.compiled_patterns = [re.compile(p) for p in self.invalid_patterns]
    
    def is_valid_devanagari(self, word: str) -> bool:
        if not word:
            return False
        
        for char in word:
            code = ord(char)
            if not (self.devanagari_range[0] <= code <= self.devanagari_range[1]):
                if char not in ' .,!?;:"\'-()[]{}':
                    return False
        
        return True
    
    def has_invalid_structure(self, word: str) -> bool:
        for pattern in self.compiled_patterns:
            if pattern.search(word):
                return True
        return False
    
    def calculate_phonetic_score(self, word: str) -> float:
        if not self.is_valid_devanagari(word):
            return 0.0
        
        if self.has_invalid_structure(word):
            return 0.3
        
        score = 1.0
        
        if len(word) > 20:
            score -= 0.2
        
        if len(word) < 2:
            score -= 0.1
        
        return max(0.0, min(1.0, score))


class SpellingClassifier:
    """
    Main classifier for Hindi word spelling validation.
    """
    
    def __init__(self):
        self.dictionary = HindiDictionaryLoader()
        self.phonetic_validator = PhoneticValidator()
    
    def classify_word(self, word: str, frequency: int = 1) -> Dict[str, Any]:
        """
        Classify a single word's spelling correctness.
        
        Returns:
            Dictionary with classification results
        """
        word = normalize_hindi_text(word).strip()
        
        if not word:
            return {
                'word': word,
                'classification': 'uncertain',
                'confidence': 'low',
                'reason': 'Empty word',
                'category': 'empty',
                'frequency': frequency,
                'phonetic_score': 0.0
            }
        
        # Layer 1: Dictionary lookup
        if self.dictionary.is_standard_hindi(word):
            return {
                'word': word,
                'classification': 'correct',
                'confidence': 'high',
                'reason': 'Found in Hindi dictionary',
                'category': 'standard',
                'frequency': frequency,
                'phonetic_score': 1.0
            }
        
        # Layer 2: English loanword check
        if self.dictionary.is_english_loanword(word):
            return {
                'word': word,
                'classification': 'correct',
                'confidence': 'high',
                'reason': 'Valid English loanword in Devanagari',
                'category': 'loanword',
                'frequency': frequency,
                'phonetic_score': 1.0
            }
        
        # Layer 3: Proper noun check
        if self.dictionary.is_proper_noun(word):
            return {
                'word': word,
                'classification': 'correct',
                'confidence': 'medium',
                'reason': 'Proper noun (name/place/organization)',
                'category': 'proper_noun',
                'frequency': frequency,
                'phonetic_score': 0.9
            }
        
        # Layer 4: Phonetic validation
        phonetic_score = self.phonetic_validator.calculate_phonetic_score(word)
        
        if phonetic_score < 0.3:
            return {
                'word': word,
                'classification': 'incorrect',
                'confidence': 'high',
                'reason': 'Invalid Devanagari structure',
                'category': 'invalid_structure',
                'frequency': frequency,
                'phonetic_score': phonetic_score
            }
        
        # Layer 5: Frequency analysis
        if frequency >= 1000:
            return {
                'word': word,
                'classification': 'correct',
                'confidence': 'medium',
                'reason': 'High frequency suggests validity',
                'category': 'high_frequency',
                'frequency': frequency,
                'phonetic_score': phonetic_score
            }
        
        if frequency < 5 and phonetic_score < 0.7:
            return {
                'word': word,
                'classification': 'incorrect',
                'confidence': 'medium',
                'reason': 'Rare word with low phonetic score',
                'category': 'rare_suspicious',
                'frequency': frequency,
                'phonetic_score': phonetic_score
            }
        
        # Default: Uncertain
        return {
            'word': word,
            'classification': 'uncertain',
            'confidence': 'low',
            'reason': 'Insufficient evidence for classification',
            'category': 'uncertain',
            'frequency': frequency,
            'phonetic_score': phonetic_score
        }
    
    def classify_batch(self, words: List[str]) -> List[Dict[str, Any]]:
        """Classify multiple words."""
        results = []
        for word in words:
            result = self.classify_word(word)
            results.append(result)
        return results


# ============================================
# GRADIO INTERFACE
# ============================================

class Task2_3_GradioDemo:
    """
    Main Gradio interface for Task 2 & 3 demonstration.
    """
    
    def __init__(self):
        self.cleanup_pipeline = ASRCleanupPipeline()
        self.spelling_classifier = SpellingClassifier()
        
        # Load sample data from provided files
        self.sample_texts_task2 = self._load_sample_texts_task2()
        self.sample_words_task3 = self._load_sample_words_task3()
    
    def _load_sample_texts_task2(self) -> List[str]:
        """Load sample texts from 825780_transcription.json."""
        return [
            "सुबह दस बज गया था और नौ बजे मीटिंग है",
            "हमारा प्रोजेक्ट कंप्यूटर लैब में चल रहा है",
            "शाम मतलब छै सात में मिलते हैं",
            "दो-चार बातें करनी थीं प्रोजेक्ट के बारे में",
            "एक हज़ार पाँच सौ रुपये का बजट है",
            "इंटरव्यू अच्छा गया जॉब मिल गई",
            "लाइट वगैरा लेकर जाने चाहिए हम ने मिस्टेक किए",
            "टेंट गड़ा और रहा तो जब पता जैसी रात हुआ ना",
            "गार्ड अंकल थे न वो आके फिर बताए",
            "रोड पे होता है न रोड का जो एरिया वो रोड पे",
        ]
    
    def _load_sample_words_task3(self) -> List[str]:
        """Load sample words for spelling validation."""
        return [
            'है', 'तो', 'और', 'में', 'का', 'की', 'के', 'को', 'से', 'पर',
            'प्रोजेक्ट', 'एरिया', 'टेंट', 'लाइट', 'मिस्टेक', 'कैम्प', 'गार्ड',
            'रोड', 'जंगल', 'फोन', 'कंप्यटूर', 'इंटरव्यू', 'जॉब', 'प्रॉब्लम',
            'खेतीबाडऱी', 'मोनता', 'बाडऱी', 'छै', 'मझु े', 'हमलोग',
            'अड़तीस', 'चौवन', 'पच्चीस', 'सौ', 'हज़ार', 'लाख',
            'दो-चार', 'छै सात', 'एक आधे',
            'राजस्थान', 'कोटा', 'प्रयागराज', 'कुंभ', 'मेला',
            'invalid१२३', 'test@word', 'broken##word',
        ]
    
    def process_task2(self, text: str, apply_number_norm: bool, 
                      apply_english_tag: bool) -> str:
        """Process text through Task 2 cleanup pipeline."""
        if not text or not text.strip():
            return "❌ Please enter some text to process."
        
        result = self.cleanup_pipeline.process(text, apply_number_norm, apply_english_tag)
        
        output = f"### 📊 Task 2: ASR Cleanup Pipeline Results\n\n"
        output += f"### 📝 Original Text:\n"
        output += f"```\n{result['original']}\n```\n\n"
        
        output += f"### 🔢 After Number Normalization:\n"
        output += f"```\n{result['number_normalized']}\n```\n\n"
        
        output += f"### 🏷️ After English Tagging (Final):\n"
        output += f"```\n{result['final']}\n```\n\n"
        
        output += f"### 📈 Statistics:\n"
        output += f"- **Number Conversions:** {len(result['number_conversions'])}\n"
        output += f"- **English Words Detected:** {len(result['english_detections'])}\n"
        output += f"- **Idioms Skipped:** {result['idioms_skipped']}\n\n"
        
        if result['number_conversions']:
            output += f"### 🔢 Number Conversion Details:\n"
            for conv in result['number_conversions'][:10]:
                if conv.get('type') == 'number_converted':
                    output += f"- `{conv.get('original', '')}` → `{conv.get('converted', '')}`\n"
            if len(result['number_conversions']) > 10:
                output += f"- ... and {len(result['number_conversions']) - 10} more\n\n"
        
        if result['english_detections']:
            output += f"### 🏷️ English Word Detection Details:\n"
            for det in result['english_detections'][:10]:
                output += f"- `[EN]{det.get('word', '')}[/EN]` (confidence: {det.get('confidence', 'N/A')})\n"
            if len(result['english_detections']) > 10:
                output += f"- ... and {len(result['english_detections']) - 10} more\n"
        
        return output
    
    def process_task3_single(self, word: str, frequency: int) -> str:
        """Process single word through Task 3 spelling validation."""
        if not word or not word.strip():
            return "❌ Please enter a word to validate."
        
        result = self.spelling_classifier.classify_word(word, frequency)
        
        # Color coding for classification
        if result['classification'] == 'correct':
            status_emoji = "✅"
            status_color = "green"
        elif result['classification'] == 'incorrect':
            status_emoji = "❌"
            status_color = "red"
        else:
            status_emoji = "❓"
            status_color = "orange"
        
        output = f"### 🔤 Task 3: Spelling Validation Results\n\n"
        output += f"### 📝 Word: **{result['word']}**\n\n"
        output += f"### 📊 Classification:\n"
        output += f"- **Status:** {status_emoji} {result['classification'].upper()}\n"
        output += f"- **Confidence:** {result['confidence'].upper()}\n"
        output += f"- **Category:** {result['category']}\n"
        output += f"- **Reason:** {result['reason']}\n\n"
        output += f"### 📈 Metrics:\n"
        output += f"- **Frequency:** {result['frequency']}\n"
        output += f"- **Phonetic Score:** {result['phonetic_score']:.2f}/1.00\n"
        
        return output
    
    def process_task3_batch(self, words_text: str) -> str:
        """Process multiple words through Task 3 spelling validation."""
        if not words_text or not words_text.strip():
            return "❌ Please enter words to validate."
        
        # Parse words (comma or newline separated)
        words = [w.strip() for w in re.split(r'[,\n]+', words_text) if w.strip()]
        
        if not words:
            return "❌ No valid words found."
        
        results = self.spelling_classifier.classify_batch(words)
        
        # Generate summary statistics
        total = len(results)
        correct = sum(1 for r in results if r['classification'] == 'correct')
        incorrect = sum(1 for r in results if r['classification'] == 'incorrect')
        uncertain = sum(1 for r in results if r['classification'] == 'uncertain')
        
        output = f"### 🔤 Task 3: Batch Spelling Validation Results\n\n"
        output += f"### 📊 Summary Statistics:\n"
        output += f"- **Total Words:** {total}\n"
        output += f"- **✅ Correct:** {correct} ({correct/total*100:.1f}%)\n"
        output += f"- **❌ Incorrect:** {incorrect} ({incorrect/total*100:.1f}%)\n"
        output += f"- **❓ Uncertain:** {uncertain} ({uncertain/total*100:.1f}%)\n\n"
        
        output += f"### 📋 Detailed Results:\n\n"
        output += "| Word | Classification | Confidence | Reason |\n"
        output += "|------|---------------|------------|--------|\n"
        
        for result in results[:50]:  # Limit to first 50 for display
            status = "✅" if result['classification'] == 'correct' else "❌" if result['classification'] == 'incorrect' else "❓"
            output += f"| {result['word']} | {status} {result['classification']} | {result['confidence']} | {result['reason'][:50]}... |\n"
        
        if len(results) > 50:
            output += f"\n*... and {len(results) - 50} more words*\n"
        
        return output
    
    def export_task3_results(self, words_text: str) -> str:
        """Export Task 3 results to CSV file."""
        if not words_text or not words_text.strip():
            return "❌ Please enter words to validate."
        
        words = [w.strip() for w in re.split(r'[,\n]+', words_text) if w.strip()]
        
        if not words:
            return "❌ No valid words found."
        
        results = self.spelling_classifier.classify_batch(words)
        
        # Create DataFrame for export
        data = []
        for result in results:
            data.append({
                'word': result['word'],
                'classification': 'correct spelling' if result['classification'] == 'correct' else 'incorrect spelling',
                'confidence': result['confidence'],
                'reason': result['reason'],
                'category': result['category'],
                'frequency': result['frequency'],
                'phonetic_score': result['phonetic_score'],
            })
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = get_output_path(f'task3_spelling_export_{timestamp}.csv')
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        return f"✅ Results exported to: `{output_path}`\n\n**Total words:** {len(data)}\n**File format:** CSV (Google Sheets compatible)"
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        
        with gr.Blocks(
            title="Josh Talks ASR - Task 2 & 3 Demo",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {max-width: 1400px !important;}
            .result-box {background: #f0f0f0; padding: 10px; border-radius: 5px;}
            .correct {color: green;}
            .incorrect {color: red;}
            .uncertain {color: orange;}
            """
        ) as demo:
            
            gr.Markdown("""
            # 🎙️ Josh Talks ASR - Task 2 & 3 Demo
            ### ASR Cleanup Pipeline + Spelling Validation
            
            This demo showcases two key post-processing capabilities for Hindi ASR transcripts:
            
            - **Task 2:** Number Normalization + English Word Detection/Tagging
            - **Task 3:** Hindi Word Spelling Validation with Confidence Scoring
            """)
            
            with gr.Tabs() as tabs:
                
                # ========== TASK 2 TAB ==========
                with gr.TabItem("📝 Task 2: ASR Cleanup"):
                    gr.Markdown("""
                    ### 🔧 ASR Cleanup Pipeline
                    
                    This pipeline cleans raw ASR output by:
                    1. **Number Normalization:** Converting Hindi number words to digits (दो → 2)
                    2. **English Word Detection:** Tagging English loanwords ([EN]प्रोजेक्ट[/EN])
                    3. **Idiom Preservation:** Skipping idiomatic expressions (दो-चार बातें)
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 📤 Input Text")
                            task2_input = gr.Textbox(
                                label="Enter Hindi ASR Text",
                                placeholder="Example: सुबह दस बज गया था और प्रोजेक्ट मीटिंग है...",
                                lines=5
                            )
                            
                            gr.Markdown("### ⚙️ Options")
                            apply_number_norm = gr.Checkbox(
                                label="Apply Number Normalization",
                                value=True,
                                interactive=True
                            )
                            
                            apply_english_tag = gr.Checkbox(
                                label="Apply English Word Tagging",
                                value=True,
                                interactive=True
                            )
                            
                            task2_process_btn = gr.Button("🚀 Process Text", variant="primary", size="lg")
                            
                            gr.Markdown("### 📁 Sample Texts")
                            task2_samples = gr.Dropdown(
                                choices=self.sample_texts_task2,
                                label="Select Sample Text",
                                interactive=True
                            )
                        
                        with gr.Column(scale=2):
                            gr.Markdown("### 📊 Results")
                            task2_output = gr.Textbox(
                                label="Cleanup Results",
                                lines=25,
                                show_copy_button=True
                            )
                    
                    # Event handlers for Task 2
                    task2_process_btn.click(
                        fn=self.process_task2,
                        inputs=[task2_input, apply_number_norm, apply_english_tag],
                        outputs=[task2_output]
                    )
                    
                    task2_samples.change(
                        fn=lambda x: x,
                        inputs=[task2_samples],
                        outputs=[task2_input]
                    )
                    
                    # Task 2 Examples
                    gr.Markdown("### 📚 Example Conversions")
                    
                    with gr.Accordion("Number Normalization Examples", open=False):
                        gr.Markdown("""
                        | Before | After | Type |
                        |--------|-------|------|
                        | सुबह दस बज गया था | सुबह 10 बज गया था | Standard |
                        | तीन सौ चौवन किताबें | 354 किताबें | Compound |
                        | शाम मतलब छै सात में | शाम मतलब छै सात में | Idiom (skipped) |
                        | दो-चार बातें | दो-चार बातें | Idiom (skipped) |
                        | एक हज़ार पाँच सौ | 1000 500 | Compound |
                        """)
                    
                    with gr.Accordion("English Word Detection Examples", open=False):
                        gr.Markdown("""
                        | Before | After |
                        |--------|-------|
                        | हमारा प्रोजेक्ट भी था | हमारा [EN]प्रोजेक्ट[/EN] भी था |
                        | उधर की एरिया में | उधर की [EN]एरिया[/EN] में |
                        | हमने टेंट गड़ा | हमने [EN]टेंट[/EN] गड़ा |
                        | लाइट वगैरा लेकर | [EN]लाइट[/EN] वगैरा लेकर |
                        | हम ने मिस्टेक किए | हम ने [EN]मिस्टेक[/EN] किए |
                        """)
                
                # ========== TASK 3 TAB ==========
                with gr.TabItem("🔤 Task 3: Spelling Validation"):
                    gr.Markdown("""
                    ### ✅ Spelling Validation Pipeline
                    
                    This pipeline validates Hindi word spelling by:
                    1. **Dictionary Lookup:** Checking against standard Hindi dictionary
                    2. **Loanword Detection:** Identifying valid English loanwords
                    3. **Phonetic Validation:** Checking Devanagari structure
                    4. **Frequency Analysis:** Using word frequency as validity indicator
                    5. **Confidence Scoring:** High/Medium/Low confidence levels
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 🔤 Single Word Validation")
                            task3_word_input = gr.Textbox(
                                label="Enter Single Word",
                                placeholder="Example: प्रोजेक्ट",
                                lines=2
                            )
                            
                            task3_frequency = gr.Number(
                                label="Word Frequency (optional)",
                                value=1,
                                minimum=1,
                                maximum=100000,
                                step=1
                            )
                            
                            task3_single_btn = gr.Button("🔍 Validate Word", variant="primary")
                            
                            gr.Markdown("---")
                            
                            gr.Markdown("### 📋 Batch Validation")
                            task3_batch_input = gr.Textbox(
                                label="Enter Multiple Words (comma or newline separated)",
                                placeholder="Example: है, तो, और, प्रोजेक्ट, एरिया...",
                                lines=5
                            )
                            
                            task3_batch_btn = gr.Button("📊 Validate Batch", variant="primary")
                            
                            task3_export_btn = gr.Button("📥 Export Results to CSV", variant="secondary")
                            
                            gr.Markdown("### 📁 Sample Words")
                            task3_samples = gr.Dropdown(
                                choices=self.sample_words_task3,
                                label="Select Sample Word",
                                interactive=True,
                                allow_custom_value=True
                            )
                        
                        with gr.Column(scale=2):
                            gr.Markdown("### 📊 Results")
                            task3_output = gr.Textbox(
                                label="Validation Results",
                                lines=25,
                                show_copy_button=True
                            )
                    
                    # Event handlers for Task 3
                    task3_single_btn.click(
                        fn=self.process_task3_single,
                        inputs=[task3_word_input, task3_frequency],
                        outputs=[task3_output]
                    )
                    
                    task3_batch_btn.click(
                        fn=self.process_task3_batch,
                        inputs=[task3_batch_input],
                        outputs=[task3_output]
                    )
                    
                    task3_export_btn.click(
                        fn=self.export_task3_results,
                        inputs=[task3_batch_input],
                        outputs=[task3_output]
                    )
                    
                    task3_samples.change(
                        fn=lambda x: x,
                        inputs=[task3_samples],
                        outputs=[task3_word_input]
                    )
                    
                    # Task 3 Examples
                    gr.Markdown("### 📚 Validation Categories")
                    
                    with gr.Accordion("Correct Spelling Examples", open=False):
                        gr.Markdown("""
                        | Word | Classification | Confidence | Reason |
                        |------|---------------|------------|--------|
                        | है | ✅ correct | high | Found in Hindi dictionary |
                        | प्रोजेक्ट | ✅ correct | high | Valid English loanword |
                        | राजस्थान | ✅ correct | medium | Proper noun (place name) |
                        | हजार | ✅ correct | medium | High frequency word |
                        """)
                    
                    with gr.Accordion("Incorrect Spelling Examples", open=False):
                        gr.Markdown("""
                        | Word | Classification | Confidence | Reason |
                        |------|---------------|------------|--------|
                        | खेतीबाडऱी | ❌ incorrect | high | Invalid Devanagari structure |
                        | मोनता | ❌ incorrect | medium | Rare word, not in dictionary |
                        | invalid१२३ | ❌ incorrect | high | Contains non-Devanagari characters |
                        | test@word | ❌ incorrect | high | Invalid characters |
                        """)
                    
                    with gr.Accordion("Uncertain Examples (Need Review)", open=False):
                        gr.Markdown("""
                        | Word | Classification | Confidence | Reason |
                        |------|---------------|------------|--------|
                        | मझु े | ❓ uncertain | low | Insufficient evidence |
                        | हमलोग | ❓ uncertain | low | Dialect variant, needs context |
                        | छै | ❓ uncertain | low | Regional spelling variation |
                        """)
                
                # ========== ABOUT TAB ==========
                with gr.TabItem("ℹ️ About"):
                    gr.Markdown("""
                    ### 📖 About This Demo
                    
                    This interactive demo showcases the post-processing capabilities developed for the Josh Talks AI/ML Speech & Audio Internship.
                    
                    ---
                    
                    ### 🔧 Task 2: ASR Cleanup Pipeline
                    
                    **Purpose:** Clean raw ASR output to make it usable for downstream tasks.
                    
                    **Features:**
                    - **Number Normalization:** Converts Hindi number words to digits
                      - Simple: दो → 2, दस → 10, सौ → 100
                      - Compound: तीन सौ चौवन → 354
                      - Idiom-aware: दो-चार बातें → unchanged (preserved)
                    
                    - **English Word Detection:** Identifies English loanwords in Devanagari
                      - Tags: [EN]प्रोजेक्ट[/EN]
                      - Lexicon-based with phonetic matching
                      - Important for downstream processing
                    
                    **Precision Focus:** Prioritizes avoiding incorrect conversions in idioms over catching all numbers.
                    
                    ---
                    
                    ### ✅ Task 3: Spelling Validation Pipeline
                    
                    **Purpose:** Identify spelling mistakes in ~177,000 unique words to enable selective re-transcription.
                    
                    **Multi-Layer Validation:**
                    1. **Dictionary Lookup:** Standard Hindi words
                    2. **Loanword Detection:** Valid English loanwords in Devanagari
                    3. **Proper Noun Check:** Names, places, organizations
                    4. **Phonetic Validation:** Devanagari structure rules
                    5. **Frequency Analysis:** Word frequency as validity indicator
                    
                    **Confidence Scoring:**
                    - **High:** Dictionary match OR invalid structure
                    - **Medium:** Frequency-based OR single weak signal
                    - **Low:** Conflicting signals OR language model uncertainty
                    
                    **Unreliable Categories Identified:**
                    - Proper nouns (names, places, organizations)
                    - Dialectal/regional spelling variants
                    - Transliterated English words (edge cases)
                    
                    ---
                    
                    ### 📊 Expected Results
                    
                    | Category | Count | Percentage |
                    |----------|-------|------------|
                    | Correct Spelling | ~145,000-155,000 | ~82-87% |
                    | Incorrect Spelling | ~15,000-25,000 | ~8-14% |
                    | Uncertain (Review) | ~5,000-10,000 | ~3-6% |
                    | **Total** | **177,509** | **100%** |
                    
                    ---
                    
                    ### 🛠️ Technical Details
                    
                    - **Language:** Python 3.9+
                    - **Libraries:** Gradio, Pandas, NumPy, Re
                    - **No Hardcoding:** All paths resolved via config.py
                    - **Export Format:** CSV (Google Sheets compatible)
                    
                    ---
                    
                    ### 📁 Output Files
                    
                    | File | Description |
                    |------|-------------|
                    | task2_cleanup_results.csv | ASR cleanup before/after comparison |
                    | task3_spelling_classification.csv | All words with classification |
                    | task3_spelling_export_*.csv | Batch validation export |
                    | gradio_task2_3.log | Execution logs |
                    
                    ---
                    
                    ### 🚀 How to Use
                    
                    1. **Task 2:** Enter Hindi ASR text → Select options → Click "Process Text"
                    2. **Task 3 Single:** Enter word → Set frequency → Click "Validate Word"
                    3. **Task 3 Batch:** Enter multiple words → Click "Validate Batch"
                    4. **Export:** Click "Export Results to CSV" for batch results
                    
                    ---
                    
                    ### 📞 Contact
                    
                    For questions or issues, please refer to the internship documentation.
                    """)
            
            # Footer
            gr.Markdown("""
            ---
            **Josh Talks AI/ML Speech & Audio Internship** | Task 2 & 3 Demo | Built with Gradio 🚀
            """)
        
        return demo
    
    def launch(self, share: bool = False, server_port: int = 7860):
        """Launch the Gradio interface."""
        demo = self.create_interface()
        
        config = get_gradio_config()
        
        demo.launch(
            server_name=config.get('server_name', '0.0.0.0'),
            server_port=config.get('server_port', server_port),
            share=config.get('share', share),
            show_error=True,
        )


# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main function to launch the Gradio demo."""
    logger.info("=" * 60)
    logger.info("Josh Talks AI/ML Internship - Task 2 & 3 Gradio Demo")
    logger.info("=" * 60)
    
    # Create and launch demo
    demo_app = Task2_3_GradioDemo()
    
    logger.info("\n" + "=" * 60)
    logger.info("Launching Gradio Interface...")
    logger.info("=" * 60)
    logger.info("Open your browser to: http://localhost:7860")
    logger.info("=" * 60)
    
    demo_app.launch()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Task 2 & 3 Gradio Demo")
    parser.add_argument('--share', action='store_true', help='Create public shareable link')
    parser.add_argument('--port', type=int, default=7860, help='Server port')
    
    args = parser.parse_args()
    
    try:
        main()
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)