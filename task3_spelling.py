"""
============================================
Josh Talks AI/ML Speech & Audio Internship
Task 3: Hindi Word Spelling Validation Pipeline
============================================

This script:
1. Loads ~177,000 unique words from Unique Words Data.xlsx
2. Classifies words as correct/incorrect spelling with confidence scores
3. Uses multi-layer validation (dictionary, frequency, phonetic, loanword)
4. Identifies unreliable word categories (proper nouns, dialects, loanwords)
5. Provides Gradio interface for manual review of low-confidence words
6. Exports results to Google Sheet format

No hardcoded paths - all data access is dynamic via config.py
"""

import os
import sys
import re
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

# Local config import
from config import (
    get_project_root,
    get_data_path,
    get_output_path,
    normalize_hindi_text,
    save_results_to_excel,
    save_results_to_csv,
    TASK3_CONFIG,
    COMMON_ENGLISH_LOANWORDS,
    GRADIO_CONFIG,
    setup_logging,
)


# ============================================
# LOGGING SETUP
# ============================================

logger = logging.getLogger(__name__)


# ============================================
# HINDI DICTIONARY & LEXICON RESOURCES
# ============================================

class HindiDictionaryLoader:
    """
    Loads and manages Hindi dictionary resources for spelling validation.
    """
    
    def __init__(self):
        self.hindi_words: Set[str] = set()
        self.english_loanwords: Set[str] = set()
        self.proper_nouns: Set[str] = set()
        self.dialect_variants: Dict[str, str] = {}
        
        self._load_resources()
    
    def _load_resources(self):
        """Load all dictionary resources."""
        # Load standard Hindi words (common words from various sources)
        self._load_standard_hindi_words()
        
        # Load English loanwords from config
        self.english_loanwords = set(COMMON_ENGLISH_LOANWORDS)
        
        # Load proper nouns (common Indian names, places, organizations)
        self._load_proper_nouns()
        
        # Load dialect variants (regional spelling variations)
        self._load_dialect_variants()
        
        logger.info(f"Loaded {len(self.hindi_words)} standard Hindi words")
        logger.info(f"Loaded {len(self.english_loanwords)} English loanwords")
        logger.info(f"Loaded {len(self.proper_nouns)} proper nouns")
        logger.info(f"Loaded {len(self.dialect_variants)} dialect variants")
    
    def _load_standard_hindi_words(self):
        """Load common Hindi words from embedded list."""
        # Common Hindi words (sample - production would load from full dictionary)
        common_hindi_words = {
            'है', 'हैं', 'था', 'थी', 'थे', 'हो', 'होता', 'होती', 'होते',
            'कर', 'किया', 'किए', 'की', 'के', 'का', 'को', 'से', 'में', 'पर',
            'और', 'या', 'तो', 'लेकिन', 'परंतु', 'मगर', 'इसलिए', 'क्योंकि',
            'क्या', 'कौन', 'कब', 'कहाँ', 'कैसे', 'कितना', 'कितनी', 'कितने',
            'यह', 'यह', 'वह', 'ये', 'वे', 'इस', 'उस', 'इन', 'उन',
            'मेरा', 'तेरा', 'हमारा', 'तुम्हारा', 'उसका', 'उनका',
            'आप', 'तुम', 'हम', 'मैं', 'वो', 'जो', 'सो', 'तो',
            'जा', 'जाता', 'जाती', 'जाते', 'आया', 'आई', 'आए', 'गया', 'गई', 'गए',
            'देख', 'देखा', 'देखी', 'देखे', 'सुन', 'सुना', 'सुनी', 'सुने',
            'बोल', 'बोला', 'बोली', 'बोले', 'कह', 'कहा', 'कही', 'कहे',
            'हो', 'होना', 'होने', 'होगा', 'होगी', 'होंगे',
            'करना', 'करने', 'करूँ', 'करे', 'करें', 'करो',
            'जाना', 'जाने', 'जाऊँ', 'जाए', 'जाएँ', 'जाओ',
            'आना', 'आने', 'आऊँ', 'आए', 'आएँ', 'आओ',
            'देना', 'देने', 'दूँ', 'दे', 'दें', 'दो',
            'लेना', 'लेने', 'लूँ', 'ले', 'लें', 'लो',
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
            'प्रोजेक्ट', 'एरिया', 'टेंट', 'लाइट', 'मिस्टेक', 'कैम्प', 'गार्ड',
            'रोड', 'जंगल', 'फोन', 'कंप्यटूर', 'इंटरव्यू', 'जॉब', 'प्रॉब्लम',
            'सॉफ्टवेयर', 'इंटरनेट', 'मोबाइल', 'लैपटॉप', 'डेस्कटॉप',
            'फीडबैक', 'ट्रेडिशनल', 'अनहेल्दी', 'हेल्दी', 'प्योर', 'हार्ट',
            'इनफॉर्मेशन', 'पार्किंग', 'फ्लोर', 'लिफ्ट', 'स्ट्रगल',
            'पसंदीदा', 'खिचड़ी', 'गिफ्टेड', 'लैंड', 'एक्सप्लोर',
            'म्यूजिक', 'डांसिंग', 'पैशन', 'एक्चुअली', 'डिश', 'डिशेस',
            'बैंड', 'बाजे', 'मूर्ति', 'गणेश', 'पूजा', 'प्रार्थना',
            'रक्षाबंधन', 'भाई', 'दूज', 'कुंभ', 'मेला', 'प्रयागराज',
            'राजस्थान', 'कोटा', 'भारत', 'भारतीय', 'हिंदी', 'अंग्रेजी',
            'शहरी', 'ग्रामीण', 'गाँव', 'शहर', 'कस्बा', 'इलाका', 'क्षेत्र',
            'जनजाति', 'जनसंख्या', 'समुदाय', 'समाज', 'संस्कृति', 'परंपरा',
            'अनुभव', 'ज्ञान', 'सीख', 'शिक्षा', 'अध्ययन', 'शोध', 'रिसर्च',
            'विकास', 'प्रगति', 'सुधार', 'परिवर्तन', 'क्रांति', 'विप्लव',
            'स्वतंत्रता', 'आजादी', 'लोकतंत्र', 'सरकार', 'राजनीति', 'नेता',
            'अर्थव्यवस्था', 'व्यापार', 'व्यवसाय', 'उद्योग', 'कारखाना',
            'कृषि', 'खेती', 'बाड़ी', 'फसल', 'बीज', 'खाद', 'सिंचाई',
            'स्वास्थ्य', 'अस्पताल', 'डॉक्टर', 'दवाई', 'इलाज', 'बीमारी',
            'खेल', 'मैदान', 'टीम', 'खिलाड़ी', 'प्रशिक्षक', 'टूर्नामेंट',
            'कला', 'संगीत', 'नृत्य', 'नाटक', 'फिल्म', 'अभिनेता', 'निर्देशक',
            'विज्ञान', 'तकनीक', 'आविष्कार', 'खोज', 'प्रयोग', 'प्रयोगशाला',
            'पर्यावरण', 'प्रकृति', 'जंगल', 'नदी', 'पहाड़', 'समुद्र', 'झील',
            'मौसम', 'गर्मी', 'सर्दी', 'बारिश', 'बादल', 'हवा', 'धूप',
            'जानवर', 'पक्षी', 'मछली', 'कीट', 'पौधा', 'पेड़', 'फूल', 'फल',
            'रंग', 'रूप', 'आकार', 'गुणवत्ता', 'मात्रा', 'गुण', 'दोष',
            'सत्य', 'झूठ', 'न्याय', 'अन्याय', 'धर्म', 'अधर्म', 'पाप', 'पुण्य',
            'आत्मा', 'शरीर', 'मन', 'बुद्धि', 'हृदय', 'चेतना', 'अवचेतन',
            'जन्म', 'मृत्यु', 'जीवन', 'मरण', 'अस्तित्व', 'अभाव',
            'समस्या', 'समाधान', 'चुनौती', 'अवसर', 'संभावना', 'निश्चय',
            'निर्णय', 'योजना', 'कार्यक्रम', 'गतिविधि', 'कार्य', 'कर्म',
            'सफलता', 'विफलता', 'जीत', 'हार', 'लाभ', 'हानि', 'फायदा', 'नुकसान',
            'सुरक्षा', 'खतरा', 'जोखिम', 'बीमा', 'रक्षा', 'हमला', 'बचाव',
            'संचार', 'संदेश', 'सूचना', 'समाचार', 'खबर', 'विज्ञापन',
            'परिवहन', 'वाहन', 'गाड़ी', 'ट्रेन', 'बस', 'हवाई', 'जहाज',
            'आवास', 'मकान', 'फ्लैट', 'बंगला', 'कमरा', 'रसोई', 'बाथरूम',
            'वस्त्र', 'कपड़े', 'शर्ट', 'पैंट', 'साड़ी', 'कुर्ता', 'जूते',
            'आभूषण', 'सोना', 'चांदी', 'हीरा', 'मोती', 'रत्न', 'गहने',
            'उत्सव', 'त्योहार', 'दीवाली', 'होली', 'ईद', 'क्रिसमस', 'बैसाखी',
            'अतिथि', 'मेहमान', 'मेजबान', 'स्वागत', 'विदाई', 'सम्मान',
            'सहयोग', 'प्रतिस्पर्धा', 'सहभागिता', 'नेतृत्व', 'अनुसरण',
            'विश्वास', 'संदेह', 'आशा', 'निराशा', 'उम्मीद', 'निराशा',
            'धैर्य', 'उत्साह', 'प्रेरणा', 'प्रोत्साहन', 'समर्थन', 'मदद',
            'कृतज्ञता', 'क्षमा', 'माफी', 'शिकायत', 'आलोचना', 'प्रशंसा',
            'समझ', 'असमझ', 'ज्ञान', 'अज्ञान', 'बुद्धिमानी', 'मूर्खता',
            'साहस', 'कायरता', 'शक्ति', 'कमजोरी', 'ताकत', 'दुर्बलता',
            'स्वच्छ', 'गंदा', 'साफ', 'मैला', 'शुद्ध', 'अशुद्ध', 'पवित्र',
            'सुंदर', 'असुंदर', 'आकर्षक', 'भद्दा', 'रुचिकर', 'बेस्वाद',
            'महत्वपूर्ण', 'अमहत्वपूर्ण', 'आवश्यक', 'अनावश्यक', 'जरूरी',
            'संभव', 'असंभव', 'संभावित', 'असंभावित', 'निश्चित', 'अनिश्चित',
            'सक्रिय', 'निष्क्रिय', 'चालू', 'बंद', 'खुला', 'बंद',
            'आंतरिक', 'बाहरी', 'ऊपरी', 'निचला', 'अंदर', 'बाहर',
            'पूर्व', 'पश्चिम', 'उत्तर', 'दक्षिण', 'पूर्वी', 'पश्चिमी',
            'प्राचीन', 'आधुनिक', 'पुरातन', 'नवीन', 'पारंपरिक', 'समकालीन',
            'व्यक्तिगत', 'सार्वजनिक', 'निजी', 'सरकारी', 'गैर-सरकारी',
            'स्थानीय', 'राष्ट्रीय', 'अंतर्राष्ट्रीय', 'वैश्विक', 'क्षेत्रीय',
            'तत्काल', 'दीर्घकालिक', 'अल्पकालिक', 'स्थायी', 'अस्थायी',
            'स्वयंसेवक', 'कर्मचारी', 'मालिक', 'नौकर', 'सेवक', 'दास',
            'अमीर', 'गरीब', 'धनी', 'निर्धन', 'मध्यम', 'उच्च', 'निम्न',
            'शिक्षित', 'अशिक्षित', 'पढ़ा-लिखा', 'अनपढ़', 'योग्य', 'अयोग्य',
            'स्वस्थ', 'अस्वस्थ', 'तंदुरुस्त', 'बीमार', 'रोगी', 'स्वस्थ',
            'जाग्रत', 'निद्रित', 'सजग', 'असावधान', 'सावधान', 'चौकस',
            'शांत', 'अशांत', 'स्थिर', 'अस्थिर', 'स्थायी', 'अस्थायी',
            'सरल', 'जटिल', 'आसान', 'कठिन', 'सुगम', 'दुर्गम', 'स्पष्ट',
            'अस्पष्ट', 'प्रकट', 'छिपा', 'खुला', 'गुप्त', 'सार्वजनिक', 'निजी',
            'स्वतंत्र', 'परतंत्र', 'आज़ाद', 'गुलाम', 'मुक्त', 'बंदी',
            'समान', 'असमान', 'बराबर', 'असमान', 'तुलनीय', 'अतुलनीय',
            'विशिष्ट', 'साधारण', 'विशेष', 'सामान्य', 'असाधारण', 'सामान्य',
            'पूर्ण', 'अपूर्ण', 'पूरा', 'अधूरा', 'संपूर्ण', 'आंशिक',
            'सही', 'गलत', 'ठीक', 'अठीक', 'उचित', 'अनुचित', 'न्यायसंगत',
            'धार्मिक', 'अधार्मिक', 'आस्तिक', 'नास्तिक', 'भक्त', 'पाखंडी',
            'मित्र', 'शत्रु', 'सखा', 'वैरी', 'सहयोगी', 'विरोधी', 'साथी',
            'परिचित', 'अपरिचित', 'ज्ञात', 'अज्ञात', 'प्रसिद्ध', 'अप्रसिद्ध',
            'लोकप्रिय', 'अलोकप्रिय', 'प्रिय', 'अप्रिय', 'प्यारा', 'नापसंद',
            'सुखद', 'दुखद', 'आनंददायक', 'कष्टदायक', 'हर्ष', 'शोक',
            'शांतिपूर्ण', 'हिंसक', 'अहिंसक', 'शांत', 'उग्र', 'कोमल', 'कठोर',
            'दयालु', 'क्रूर', 'करुणा', 'निर्दय', 'मानवतावादी', 'अमानवीय',
            'ईमानदार', 'बेईमान', 'सच्चा', 'झूठा', 'विश्वसनीय', 'अविश्वसनीय',
            'न्यायप्रिय', 'अन्यायप्रिय', 'निष्पक्ष', 'पक्षपाती', 'तटस्थ',
            'उदार', 'कंजूस', 'दानवीर', 'लोभी', 'त्यागी', 'स्वार्थी',
            'विनम्र', 'घमंडी', 'नम्र', 'अहंकारी', 'गर्वीला', 'निर्मम',
            'सहनशील', 'असहनशील', 'धैर्यवान', 'चिड़चिड़ा', 'शांत', 'कोपिष्ठ',
            'मिलनसार', 'एकाकी', 'सामाजिक', 'असामाजिक', 'बातूनी', 'मौन',
            'हास्यप्रिय', 'गंभीर', 'मज़ाकिया', 'रुक्ष', 'रसिक', 'नीरस',
            'रोमांचक', 'बोरिंग', 'दिलचस्प', 'अरुचिकर', 'आकर्षक', 'विकर्षक',
            'प्रभावी', 'अप्रभावी', 'सफल', 'विफल', 'कारगर', 'बेअसर',
            'कुशल', 'अकुशल', 'दक्ष', 'अदक्ष', 'योग्य', 'अयोग्य', 'सक्षम',
            'असमर्थ', 'शक्तिशाली', 'दुर्बल', 'प्रभावशाली', 'निष्प्रभाव',
            'बुद्धिमान', 'मूर्ख', 'चतुर', 'भोला', 'चालाक', 'सीधा', 'धूर्त',
            'कलात्मक', 'अकलात्मक', 'रचनात्मक', 'अरचनात्मक', 'सृजनशील',
            'वैज्ञानिक', 'अवैज्ञानिक', 'तार्किक', 'अतार्किक', 'युक्तिपूर्ण',
            'व्यावहारिक', 'अव्यावहारिक', 'प्रयोगात्मक', 'सैद्धांतिक',
            'आध्यात्मिक', 'भौतिकवादी', 'लौकिक', 'अलौकिक', 'पारलौकिक',
            'नैतिक', 'अनैतिक', 'सदाचारी', 'दुराचारी', 'पुण्यात्मा', 'पापी',
            'परोपकारी', 'स्वार्थी', 'सेवक', 'भोगी', 'त्यागी', 'लोभी',
            'देशभक्त', 'देशद्रोही', 'राष्ट्रवादी', 'अंतर्राष्ट्रीयवादी',
            'पर्यावरणप्रेमी', 'प्रदूषक', 'संरक्षक', 'विनाशक', 'रक्षक',
            'मानवतावादी', 'संप्रदायवादी', 'सर्वाहारी', 'शाकाहारी', 'मांसाहारी',
            'पारंपरिक', 'आधुनिकतावादी', 'रूढ़िवादी', 'प्रगतिशील', 'क्रांतिकारी',
            'सुधारवादी', 'यथास्थितिवादी', 'विचारशील', 'अविचारशील', 'बुद्धिजीवी',
            'कर्मठ', 'आलसी', 'मेहनती', 'कायर', 'साहसी', 'निडर', 'डरपोक',
            'आशावादी', 'निराशावादी', 'आश्वस्त', 'चिंतित', 'चिंतामुक्त',
            'संतुष्ट', 'असंतुष्ट', 'तृप्त', 'भूखा', 'प्यासा', 'तृप्तिदायक',
            'आरामदायक', 'असुविधाजनक', 'सुखदायक', 'कष्टप्रद', 'सुविधाजनक',
            'लाभदायक', 'हानिकारक', 'फायदेमंद', 'नुकसानदेह', 'उपयोगी',
            'अनुपयोगी', 'कारगर', 'बेकार', 'मूल्यवान', 'अमूल्य', 'कीमती',
            'सस्ता', 'महंगा', 'किफायती', 'बर्बाद', 'मितव्ययी', 'फिजूलखर्च',
            'समयपाबंद', 'असमयपाबंद', 'पunctual', 'लेट', 'जल्दी', 'धीमा',
            'तेज', 'गतिमान', 'स्थिर', 'चंचल', 'स्थिरचित्त', 'विचलित',
            'एकाग्र', 'विकेंद्रित', 'ध्यानवान', 'असावधान', 'सतर्क', 'निश्चिंत',
            'जिज्ञासु', 'उदासीन', 'उत्सुक', 'अरुचि', 'रुचि', 'विरुचि',
            'लगन', 'वैराग्य', 'आसक्ति', 'अनासक्ति', 'बंधन', 'मुक्ति', 'बंधा',
            'स्वामी', 'सेवक', 'मालिक', 'नौकर', 'राजा', 'प्रजा', 'शासक',
            'शासित', 'नेता', 'अनुयायी', 'गुरु', 'शिष्य', 'आचार्य', 'विद्यार्थी',
            'अध्यापक', 'छात्र', 'प्रोफेसर', 'लेक्चरर', 'शोधार्थी', 'विद्वान',
            'पंडित', 'मौलवी', 'पादरी', 'भिक्षु', 'साधु', 'संत', 'योगी',
            'तपस्वी', 'गृहस्थ', 'वानप्रस्थ', 'संन्यासी', 'ब्रह्मचारी', 'गृही',
            'पति', 'पत्नी', 'वर', 'वधू', 'दूल्हा', 'दुल्हन', 'प्रेमी', 'प्रेमिका',
            'पुत्र', 'पुत्री', 'बेटा', 'बेटी', 'बच्चा', 'शिशु', 'किशोर', 'युवा',
            'वृद्ध', 'बूढ़ा', 'जवान', 'मध्यम', 'किशोरी', 'युवती', 'वृद्धा',
            'पितामह', 'पितामही', 'दादा', 'दादी', 'नाना', 'नानी', 'पोता', 'पोती',
            'भतीजा', 'भतीजी', 'भांजा', 'भांजी', 'चाचा', 'चाची', 'ताऊ', 'ताई',
            'मामा', 'मामी', 'फूफा', 'फूफी', 'बुआ', 'फूफी', 'साला', 'साली',
            'बहनोई', 'देवर', 'ननद', 'जिजा', 'ससुर', 'सास', 'दामाद', 'बहू',
            'पड़ोसी', 'मित्र', 'सखा', 'यार', 'दोस्त', 'संगी', 'साथी', 'हमराही',
            'सहयात्री', 'सहकर्मी', 'सहपाठी', 'सहयोगी', 'प्रतिद्वंद्वी', 'प्रतिस्पर्धी',
            'दुश्मन', 'शत्रु', 'वैरी', 'अरी', 'बाधा', 'विघ्न', 'रुकावट', 'बाधक',
            'सहायक', 'सहयोगी', 'मददगार', 'रक्षक', 'संरक्षक', 'अभिभावक', 'पालक',
            'पोषक', 'दाता', 'दानवीर', 'परोपकारी', 'सेवक', 'सेविका', 'कर्मचारी',
            'अधिकारी', 'प्रशासक', 'प्रबंधक', 'निर्देशक', 'अध्यक्ष', 'सचिव',
            'कोषाध्यक्ष', 'सदस्य', 'सांसेद', 'मंत्री', 'मुख्यमंत्री', 'प्रधानमंत्री',
            'राष्ट्रपति', 'उपराष्ट्रपति', 'गवर्नर', 'लेफ्टिनेंट', 'कलेक्टर',
            'मैजिस्ट्रेट', 'जज', 'वकील', 'अधिवक्ता', 'पैरवीकार', 'न्यायाधीश',
            'पुलिस', 'सिपाही', 'कांस्टेबल', 'इंस्पेक्टर', 'एसआई', 'डीएसपी',
            'एसपी', 'आईजी', 'डीजी', 'सेना', 'फौज', 'सैनिक', 'सिपाही', 'जवान',
            'अफसर', 'जनरल', 'मेजर', 'कर्नल', 'कैप्टन', 'लेफ्टिनेंट', 'सुबेदार',
            'हवलदार', 'नायक', 'लांस', 'सिपाही', 'कमांडो', 'कमांडर', 'चीफ',
            'एडमिरल', 'कमोडोर', 'पायलट', 'नेविगेटर', 'ड्राइवर', 'कंडक्टर',
            'गार्ड', 'टिकट', 'चेकर', 'सुपरवाइजर', 'मैनेजर', 'सुपरवाइजर',
            'फोरमैन', 'मिस्त्री', 'कारिगर', 'श्रमिक', 'मजदूर', 'कर्मकार',
            'किसान', 'खेतिहर', 'माली', 'बढ़ई', 'लोहार', 'कुम्हार', 'जुलाहा',
            'दर्जी', 'नाई', 'धोबी', 'मोची', 'सुनार', 'कसाई', 'मछुआरा',
            'चरवाहा', 'गडरिया', 'अहिर', 'गुज्जर', 'जाट', 'राजपूत', 'ठाकुर',
            'ब्राह्मण', 'क्षत्रिय', 'वैश्य', 'शूद्र', 'हरिजन', 'दलित', 'आदिवासी',
            'जनजाति', 'अनुसूचित', 'पिछड़ा', 'अग्र', 'उच्च', 'निम्न', 'मध्यम',
            'वर्ग', 'श्रेणी', 'वर्गीकरण', 'श्रेणीकरण', 'विभाजन', 'खंड', 'भाग',
            'अंश', 'टुकड़ा', 'हिस्सा', 'भाग', 'पक्ष', 'कोण', 'दिशा', 'ओर',
            'तरफ', 'जानिब', 'सू', 'अभिमुख', 'सम्मुख', 'सामने', 'पीछे', 'पार्श्व',
            'बगल', 'किनारा', 'सीमा', 'हद', 'सरहद', 'बॉर्डर', 'अंतर्राष्ट्रीय',
            'राष्ट्रीय', 'प्रांतीय', 'जिला', 'तहसील', 'ब्लॉक', 'ग्राम', 'पंचायत',
            'नगर', 'नगरपालिका', 'नगर निगम', 'महानगर', 'राजधानी', 'राज्य',
            'संघ', 'केंद्र', 'सरकार', 'प्रशासन', 'तंत्र', 'व्यवस्था', 'प्रणाली',
            'नीति', 'नियम', 'कानून', 'विधि', 'अधिनियम', 'संविधान', 'कानून',
            'न्याय', 'अदालत', 'मुकदमा', 'वादी', 'प्रतिवादी', 'वकील', 'पैरवी',
            'दलील', 'तर्क', 'साक्ष्य', 'गवाह', 'सबूत', 'प्रमाण', 'साक्ष्य',
            'फैसला', 'निर्णय', 'आदेश', 'हुक्म', 'फरमान', 'घोषणा', 'विज्ञप्ति',
            'सूचना', 'संदेश', 'संकेत', 'इशारा', 'संकेत', 'संकेत', 'संकेत',
        }
        
        self.hindi_words.update(common_hindi_words)
    
    def _load_proper_nouns(self):
        """Load common proper nouns (names, places, organizations)."""
        proper_nouns = {
            # Indian names
            'राम', 'श्याम', 'कृष्ण', 'अर्जुन', 'भीम', 'युधिष्ठिर', 'नकुल', 'सहदेव',
            'सीता', 'द्रौपदी', 'कुंती', 'गांधारी', 'सुभद्रा', 'रुक्मिणी', 'सत्यभामा',
            'राधा', 'मीरा', 'सीता', 'उर्मिला', 'मांडवी', 'श्रुतकीर्ति',
            'भरत', 'लक्ष्मण', 'शत्रुघ्न', 'दशरथ', 'कौशल्या', 'कैकेयी', 'सुमित्रा',
            'रावण', 'कुंभकर्ण', 'विभीषण', 'इंद्रजीत', 'अंगद', 'सुग्रीव', 'बाली',
            'हनुमान', 'जाम्बवंत', 'नल', 'नील', 'सुषेण', 'गंधमादन',
            
            # Places
            'भारत', 'हिंदुस्तान', 'इंडिया', 'दिल्ली', 'मुंबई', 'कोलकाता', 'चेन्नई',
            'बेंगलुरु', 'हैदराबाद', 'पुणे', 'अहमदाबाद', 'जयपुर', 'लखनऊ', 'कानपुर',
            'नागपुर', 'इंदौर', 'थाने', 'भोपाल', 'विशाखापत्तनम', 'पटना', 'वडोदरा',
            'गाजियाबाद', 'लुधियाना', 'आगरा', 'नासिक', 'फरीदाबाद', 'मेरठ', 'राजकोट',
            'कल्याण', 'वाराणसी', 'श्रीनगर', 'औरंगाबाद', 'धनबाद', 'अमृतसर', 'नवीमुंबई',
            'इलाहाबाद', 'रांची', 'हावड़ा', 'कोयंबतूर', 'जबलपुर', 'गुलबर्गा', 'गवालियर',
            'विजयवाड़ा', 'जोधपुर', 'मदुरै', 'रायपुर', 'कोटा', 'गुवाहाटी', 'चंडीगढ़',
            'सोलन', 'शिमला', 'धर्मशाला', 'मनाली', 'कुल्लू', 'कांगड़ा', 'चंबा',
            'उदयपुर', 'जैसलमेर', 'जोधपुर', 'बीकानेर', 'कोटा', 'अजमेर', 'पुष्कर',
            'आगरा', 'मथुरा', 'वृंदावन', 'अयोध्या', 'वाराणसी', 'हरिद्वार', 'ऋषिकेश',
            'प्रयागराज', 'इलाहाबाद', 'कानपुर', 'लखनऊ', 'वाराणसी', 'गया', 'पटना',
            'राजस्थान', 'गुजरात', 'महाराष्ट्र', 'कर्नाटक', 'तमिलनाडु', 'केरल',
            'आंध्रप्रदेश', 'तेलंगाना', 'ओडिशा', 'पश्चिमबंगाल', 'बिहार', 'झारखंड',
            'छत्तीसगढ़', 'मध्यप्रदेश', 'उत्तरप्रदेश', 'उत्तराखंड', 'हिमाचल', 'पंजाब',
            'हरियाणा', 'जम्मू', 'कश्मीर', 'लद्दाख', 'अरुणाचल', 'असम', 'मेघालय',
            'मणिपुर', 'मिजोरम', 'नागालैंड', 'सिक्किम', 'त्रिपुरा', 'गोवा', 'दमन',
            'दियू', 'दादरा', 'नगरहवेली', 'पुडुचेरी', 'लक्षद्वीप', 'अंडमान',
            
            # Organizations
            'जोश', 'टॉक्स', 'जोशटॉक्स', 'Josh', 'Talks', 'Google', 'Facebook',
            'WhatsApp', 'Instagram', 'YouTube', 'Twitter', 'LinkedIn', 'TikTok',
            'Amazon', 'Flipkart', 'Paytm', 'PhonePe', 'Zomato', 'Swiggy',
            'Ola', 'Uber', 'Zoho', 'TCS', 'Infosys', 'Wipro', 'HCL', 'TechM',
            'Reliance', 'Tata', 'Birla', 'Ambani', 'Adani', 'Mahindra', 'Bajaj',
            'Hero', 'Honda', 'Maruti', 'Hyundai', 'Toyota', 'Ford', 'BMW',
            'Mercedes', 'Audi', 'Volkswagen', 'Nissan', 'Renault', 'Suzuki',
            'Microsoft', 'Apple', 'Samsung', 'Sony', 'LG', 'Panasonic', 'Philips',
            'Canon', 'Nikon', 'HP', 'Dell', 'Lenovo', 'Asus', 'Acer', 'Intel',
            'AMD', 'Nvidia', 'Qualcomm', 'Snapdragon', 'MediaTek', 'ARM',
            'Netflix', 'Prime', 'Hotstar', 'JioCinema', 'SonyLIV', 'Voot',
            'Zee5', 'ALTBalaji', 'MXPlayer', 'Disney', 'HBO', 'ESPN',
            'CNN', 'BBC', 'NDTV', 'AajTak', 'ABP', 'ZeeNews', 'TimesNow',
            'Republic', 'IndiaTV', 'News18', 'News24', 'Sudarshan', 'TV9',
            'Hindustan', 'DainikJagran', 'AmarUjala', 'NavbharatTimes', 'DainikBhaskar',
            'RajasthanPatrika', 'PrabhatKhabar', 'Jansatta', 'IndianExpress', 'TheHindu',
            'TimesOfIndia', 'HindustanTimes', 'TheTelegraph', 'TheStatesman', 'MidDay',
            'Mint', 'BusinessStandard', 'FinancialExpress', 'EconomicTimes', 'MoneyControl',
        }
        
        self.proper_nouns.update(proper_nouns)
    
    def _load_dialect_variants(self):
        """Load dialect/regional spelling variants."""
        self.dialect_variants = {
            # Standard → Dialect variants
            'छह': ['छै', 'छः'],
            'मुझे': ['मझु े', 'मुझको', 'मोको'],
            'हम': ['हमलोग', 'हमरा', 'हमने'],
            'तुम': ['तुमलोग', 'तुमरा', 'तुमने'],
            'है': ['हय', 'हे', 'ह'],
            'हैं': ['हैं', 'हय', 'ह'],
            'था': ['तھا', 'तह'],
            'थी': ['तھی', 'तहि'],
            'थे': ['تھے', 'तहे'],
            'करना': ['करना', 'करना'],
            'जाता': ['जाता', 'जात'],
            'जाती': ['जाती', 'जाति'],
            'जाते': ['जाते', 'जात'],
            'आता': ['आता', 'आत'],
            'आती': ['आती', 'आति'],
            'आते': ['आते', 'आत'],
            'होता': ['होता', 'होत'],
            'होती': ['होती', 'होति'],
            'होते': ['होते', 'होत'],
            'लोग': ['लोग', 'लोगन', 'लोक'],
            'वाला': ['वाला', 'आला', 'बाला'],
            'वाली': ['वाली', 'आली', 'बाली'],
            'वाले': ['वाले', 'आले', 'बाले'],
        }
    
    def is_standard_hindi(self, word: str) -> bool:
        """Check if word is in standard Hindi dictionary."""
        return word in self.hindi_words
    
    def is_english_loanword(self, word: str) -> bool:
        """Check if word is a known English loanword."""
        return word in self.english_loanwords
    
    def is_proper_noun(self, word: str) -> bool:
        """Check if word is a proper noun."""
        return word in self.proper_nouns
    
    def is_dialect_variant(self, word: str) -> Tuple[bool, Optional[str]]:
        """
        Check if word is a dialect variant of a standard word.
        
        Returns:
            Tuple of (is_variant, standard_form)
        """
        for standard, variants in self.dialect_variants.items():
            if word in variants:
                return (True, standard)
        return (False, None)


# ============================================
# PHONETIC VALIDATION MODULE
# ============================================

class PhoneticValidator:
    """
    Validates Devanagari word structure using phonetic rules.
    """
    
    def __init__(self):
        # Valid Devanagari character ranges
        self.devanagari_range = (0x0900, 0x097F)
        
        # Common invalid patterns
        self.invalid_patterns = [
            r'[ाीीूूेेैैोोौौ]{2,}',  # Repeated vowel signs
            r'[क-ह]{4,}',  # Too many consecutive consonants
            r'[़़]{2,}',  # Repeated nukta
            r'[ंंं]{2,}',  # Repeated anusvara
            r'[ःः]{2,}',  # Repeated visarga
        ]
        
        # Compile regex patterns
        self.compiled_patterns = [re.compile(p) for p in self.invalid_patterns]
    
    def is_valid_devanagari(self, word: str) -> bool:
        """
        Check if word contains valid Devanagari characters.
        
        Args:
            word: Input word
        
        Returns:
            True if valid Devanagari, False otherwise
        """
        if not word:
            return False
        
        for char in word:
            code = ord(char)
            # Check if character is in Devanagari range
            if not (self.devanagari_range[0] <= code <= self.devanagari_range[1]):
                # Allow some common punctuation
                if char not in ' .,!?;:"\'-()[]{}':
                    return False
        
        return True
    
    def has_invalid_structure(self, word: str) -> bool:
        """
        Check if word has invalid phonetic structure.
        
        Args:
            word: Input word
        
        Returns:
            True if structure is invalid, False otherwise
        """
        for pattern in self.compiled_patterns:
            if pattern.search(word):
                return True
        
        return False
    
    def calculate_phonetic_score(self, word: str) -> float:
        """
        Calculate a phonetic validity score for the word.
        
        Args:
            word: Input word
        
        Returns:
            Score between 0.0 (invalid) and 1.0 (valid)
        """
        if not self.is_valid_devanagari(word):
            return 0.0
        
        if self.has_invalid_structure(word):
            return 0.3
        
        # Basic scoring based on character patterns
        score = 1.0
        
        # Penalize very long words (likely errors)
        if len(word) > 20:
            score -= 0.2
        
        # Penalize very short words (might be fragments)
        if len(word) < 2:
            score -= 0.1
        
        return max(0.0, min(1.0, score))


# ============================================
# FREQUENCY ANALYSIS MODULE
# ============================================

class FrequencyAnalyzer:
    """
    Analyzes word frequency in the dataset for spelling validation.
    """
    
    def __init__(self, frequency_thresholds: Dict[str, int] = None):
        if frequency_thresholds is None:
            frequency_thresholds = TASK3_CONFIG.get('frequency_thresholds', {
                'high_freq': 1000,
                'low_freq': 5,
            })
        
        self.high_freq_threshold = frequency_thresholds['high_freq']
        self.low_freq_threshold = frequency_thresholds['low_freq']
        self.word_frequencies: Counter = Counter()
    
    def load_frequencies_from_dataset(self, df: pd.DataFrame) -> None:
        """
        Load word frequencies from the unique words dataset.
        
        Args:
            df: DataFrame with word frequency data (if available)
        """
        # If dataset has frequency column, use it
        if 'frequency' in df.columns:
            for _, row in df.iterrows():
                self.word_frequencies[row['word']] = row['frequency']
        else:
            # Otherwise, assume uniform frequency (will be updated if full corpus available)
            for word in df['word'].dropna().unique():
                self.word_frequencies[word] = 1
    
    def get_frequency_category(self, word: str) -> str:
        """
        Categorize word by frequency.
        
        Args:
            word: Input word
        
        Returns:
            Category: 'high', 'medium', or 'low'
        """
        freq = self.word_frequencies.get(word, 0)
        
        if freq >= self.high_freq_threshold:
            return 'high'
        elif freq >= self.low_freq_threshold:
            return 'medium'
        else:
            return 'low'
    
    def get_frequency_score(self, word: str) -> float:
        """
        Calculate frequency-based confidence score.
        
        Args:
            word: Input word
        
        Returns:
            Score between 0.0 (rare) and 1.0 (common)
        """
        freq = self.word_frequencies.get(word, 0)
        
        if freq >= self.high_freq_threshold:
            return 1.0
        elif freq >= self.low_freq_threshold:
            return 0.7
        elif freq > 0:
            return 0.4
        else:
            return 0.2


# ============================================
# SPELLING CLASSIFIER
# ============================================

@dataclass
class WordClassification:
    """Represents classification result for a single word."""
    word: str
    classification: str  # 'correct', 'incorrect', 'uncertain'
    confidence: str  # 'high', 'medium', 'low'
    reason: str
    category: Optional[str] = None  # 'standard', 'loanword', 'proper_noun', 'dialect'
    frequency: int = 0
    phonetic_score: float = 0.0


class SpellingClassifier:
    """
    Main classifier for Hindi word spelling validation.
    """
    
    def __init__(self):
        self.dictionary = HindiDictionaryLoader()
        self.phonetic_validator = PhoneticValidator()
        self.frequency_analyzer = FrequencyAnalyzer()
        
        self.classifications: List[WordClassification] = []
        self.unreliable_categories: Dict[str, List[str]] = defaultdict(list)
    
    def classify_word(self, word: str, frequency: int = 1) -> WordClassification:
        """
        Classify a single word's spelling correctness.
        
        Args:
            word: Input Hindi word
            frequency: Word frequency in corpus
        
        Returns:
            WordClassification object
        """
        word = normalize_hindi_text(word).strip()
        
        if not word:
            return WordClassification(
                word=word,
                classification='uncertain',
                confidence='low',
                reason='Empty word'
            )
        
        # Layer 1: Dictionary lookup (highest confidence)
        if self.dictionary.is_standard_hindi(word):
            return WordClassification(
                word=word,
                classification='correct',
                confidence='high',
                reason='Found in Hindi dictionary',
                category='standard',
                frequency=frequency,
                phonetic_score=1.0
            )
        
        # Layer 2: English loanword check
        if self.dictionary.is_english_loanword(word):
            return WordClassification(
                word=word,
                classification='correct',
                confidence='high',
                reason='Valid English loanword in Devanagari',
                category='loanword',
                frequency=frequency,
                phonetic_score=1.0
            )
        
        # Layer 3: Proper noun check
        if self.dictionary.is_proper_noun(word):
            return WordClassification(
                word=word,
                classification='correct',
                confidence='medium',
                reason='Proper noun (name/place/organization)',
                category='proper_noun',
                frequency=frequency,
                phonetic_score=0.9
            )
        
        # Layer 4: Dialect variant check
        is_variant, standard_form = self.dictionary.is_dialect_variant(word)
        if is_variant:
            return WordClassification(
                word=word,
                classification='correct',
                confidence='medium',
                reason=f'Dialect variant of "{standard_form}"',
                category='dialect',
                frequency=frequency,
                phonetic_score=0.8
            )
        
        # Layer 5: Phonetic validation
        phonetic_score = self.phonetic_validator.calculate_phonetic_score(word)
        
        if phonetic_score < 0.3:
            return WordClassification(
                word=word,
                classification='incorrect',
                confidence='high',
                reason='Invalid Devanagari structure',
                category='invalid_structure',
                frequency=frequency,
                phonetic_score=phonetic_score
            )
        
        # Layer 6: Frequency analysis
        freq_category = self.frequency_analyzer.get_frequency_category(word)
        freq_score = self.frequency_analyzer.get_frequency_score(word)
        
        if freq_category == 'high':
            return WordClassification(
                word=word,
                classification='correct',
                confidence='medium',
                reason='High frequency suggests validity',
                category='high_frequency',
                frequency=frequency,
                phonetic_score=phonetic_score
            )
        
        if freq_category == 'low' and phonetic_score < 0.7:
            return WordClassification(
                word=word,
                classification='incorrect',
                confidence='medium',
                reason='Rare word with low phonetic score',
                category='rare_suspicious',
                frequency=frequency,
                phonetic_score=phonetic_score
            )
        
        # Default: Uncertain (needs manual review)
        return WordClassification(
            word=word,
            classification='uncertain',
            confidence='low',
            reason='Insufficient evidence for classification',
            category='uncertain',
            frequency=frequency,
            phonetic_score=phonetic_score
        )
    
    def classify_batch(self, words: List[str], frequencies: Optional[List[int]] = None) -> List[WordClassification]:
        """
        Classify multiple words.
        
        Args:
            words: List of words to classify
            frequencies: Optional list of frequencies (same length as words)
        
        Returns:
            List of WordClassification objects
        """
        if frequencies is None:
            frequencies = [1] * len(words)
        
        classifications = []
        for word, freq in tqdm(zip(words, frequencies), total=len(words), desc="Classifying words"):
            classification = self.classify_word(word, freq)
            classifications.append(classification)
            self.classifications.append(classification)
        
        return classifications
    
    def get_low_confidence_words(self, n: int = 50) -> List[WordClassification]:
        """
        Get sample of low-confidence words for manual review.
        
        Args:
            n: Number of words to sample
        
        Returns:
            List of low-confidence classifications
        """
        low_conf = [c for c in self.classifications if c.confidence == 'low']
        
        # Stratified sampling
        if len(low_conf) <= n:
            return low_conf
        
        # Sample from different categories
        uncertain = [c for c in low_conf if c.classification == 'uncertain']
        incorrect = [c for c in low_conf if c.classification == 'incorrect']
        
        sample_size = min(n, len(low_conf))
        return random.sample(low_conf, sample_size)
    
    def identify_unreliable_categories(self) -> Dict[str, List[str]]:
        """
        Identify word categories where the system is unreliable.
        
        Returns:
            Dictionary of category → example words
        """
        # Analyze misclassifications (would need ground truth for full analysis)
        # For now, identify categories with high uncertainty
        
        category_examples = defaultdict(list)
        
        for classification in self.classifications:
            if classification.confidence == 'low':
                if classification.category not in category_examples:
                    category_examples[classification.category] = []
                
                if len(category_examples[classification.category]) < 10:
                    category_examples[classification.category].append(classification.word)
        
        self.unreliable_categories = dict(category_examples)
        
        return self.unreliable_categories
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get classification statistics.
        
        Returns:
            Dictionary of statistics
        """
        if not self.classifications:
            return {}
        
        total = len(self.classifications)
        
        by_classification = Counter(c.classification for c in self.classifications)
        by_confidence = Counter(c.confidence for c in self.classifications)
        by_category = Counter(c.category for c in self.classifications)
        
        return {
            'total_words': total,
            'correct_count': by_classification.get('correct', 0),
            'incorrect_count': by_classification.get('incorrect', 0),
            'uncertain_count': by_classification.get('uncertain', 0),
            'correct_percentage': (by_classification.get('correct', 0) / total * 100) if total > 0 else 0,
            'incorrect_percentage': (by_classification.get('incorrect', 0) / total * 100) if total > 0 else 0,
            'high_confidence_count': by_confidence.get('high', 0),
            'medium_confidence_count': by_confidence.get('medium', 0),
            'low_confidence_count': by_confidence.get('low', 0),
            'by_category': dict(by_category),
        }


# ============================================
# GRADIO INTERFACE FOR MANUAL REVIEW
# ============================================

def create_gradio_review_interface(classifier: SpellingClassifier) -> None:
    """
    Create Gradio interface for manual review of low-confidence words.
    
    Args:
        classifier: SpellingClassifier instance
    """
    try:
        import gradio as gr
    except ImportError:
        logger.error("Gradio not installed. Run: pip install gradio")
        return
    
    # Get low-confidence words for review
    low_conf_words = classifier.get_low_confidence_words(n=50)
    review_queue = list(low_conf_words)
    current_index = [0]
    review_results = []
    
    def get_current_word() -> Dict[str, Any]:
        """Get current word for review."""
        if current_index[0] >= len(review_queue):
            return {
                'word': 'Review Complete!',
                'classification': '',
                'confidence': '',
                'reason': f'All {len(review_queue)} words reviewed.',
                'progress': f'{len(review_queue)}/{len(review_queue)}'
            }
        
        item = review_queue[current_index[0]]
        return {
            'word': item.word,
            'classification': item.classification,
            'confidence': item.confidence,
            'reason': item.reason,
            'progress': f'{current_index[0] + 1}/{len(review_queue)}'
        }
    
    def submit_annotation(annotation: str) -> Dict[str, Any]:
        """Submit annotation and move to next word."""
        if current_index[0] < len(review_queue):
            item = review_queue[current_index[0]]
            review_results.append({
                'word': item.word,
                'system_classification': item.classification,
                'system_confidence': item.confidence,
                'system_reason': item.reason,
                'human_annotation': annotation,
            })
        
        current_index[0] += 1
        return get_current_word()
    
    def reset_review() -> Dict[str, Any]:
        """Reset review to beginning."""
        current_index[0] = 0
        review_results.clear()
        return get_current_word()
    
    def export_results() -> str:
        """Export review results to CSV."""
        if not review_results:
            return "No results to export."
        
        output_path = get_output_path('task3_manual_review_results.csv')
        df = pd.DataFrame(review_results)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        return f"Results exported to: {output_path}"
    
    with gr.Blocks(title="Josh Talks Spelling Review", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔤 Hindi Word Spelling Manual Review Interface")
        gr.Markdown("""
        Review low-confidence word classifications from the automated system.
        Select the correct classification for each word to improve the system.
        """)
        
        current = get_current_word()
        
        with gr.Row():
            with gr.Column(scale=2):
                word_display = gr.Textbox(
                    label="Word",
                    value=current['word'],
                    interactive=False
                )
                
                classification_display = gr.Textbox(
                    label="System Classification",
                    value=current['classification'],
                    interactive=False
                )
                
                confidence_display = gr.Textbox(
                    label="System Confidence",
                    value=current['confidence'],
                    interactive=False
                )
                
                reason_display = gr.Textbox(
                    label="System Reason",
                    value=current['reason'],
                    interactive=False,
                    lines=2
                )
                
                progress_display = gr.Textbox(
                    label="Progress",
                    value=current['progress'],
                    interactive=False
                )
            
            with gr.Column(scale=2):
                annotation = gr.Radio(
                    choices=[
                        ("✅ Correct Spelling", "correct"),
                        ("❌ Incorrect Spelling", "incorrect"),
                        ("❓ Uncertain/Needs Context", "uncertain")
                    ],
                    label="Your Annotation",
                    value="correct"
                )
                
                submit_btn = gr.Button("Submit & Next →", variant="primary")
                reset_btn = gr.Button("🔄 Reset Review", variant="secondary")
                export_btn = gr.Button("📥 Export Results", variant="secondary")
                
                export_output = gr.Textbox(label="Export Status", interactive=False)
        
        # Initialize with first word
        def update_display():
            current = get_current_word()
            return (
                current['word'],
                current['classification'],
                current['confidence'],
                current['reason'],
                current['progress']
            )
        
        submit_btn.click(
            fn=submit_annotation,
            inputs=[annotation],
            outputs=[word_display, classification_display, confidence_display, reason_display, progress_display]
        ).then(
            fn=update_display,
            outputs=[word_display, classification_display, confidence_display, reason_display, progress_display]
        )
        
        reset_btn.click(
            fn=reset_review,
            outputs=[word_display, classification_display, confidence_display, reason_display, progress_display]
        )
        
        export_btn.click(
            fn=export_results,
            outputs=[export_output]
        )
    
    # Launch Gradio
    config = get_gradio_config()
    demo.launch(
        server_name=config.get('server_name', '0.0.0.0'),
        server_port=config.get('server_port', 7860),
        share=config.get('share', False),
    )


# ============================================
# RESULTS EXPORT
# ============================================

def export_classification_results(
    classifications: List[WordClassification],
    output_path: Optional[Path] = None
) -> Path:
    """
    Export classification results to Google Sheet format.
    
    Args:
        classifications: List of WordClassification objects
        output_path: Output file path
    
    Returns:
        Path to saved file
    """
    if output_path is None:
        output_path = get_output_path('task3_spelling_classification.csv')
    
    # Create DataFrame matching required format
    data = []
    for c in classifications:
        data.append({
            'word': c.word,
            'classification': 'correct spelling' if c.classification == 'correct' else 'incorrect spelling',
            'confidence': c.confidence,
            'reason': c.reason,
            'category': c.category,
            'frequency': c.frequency,
            'phonetic_score': c.phonetic_score,
        })
    
    df = pd.DataFrame(data)
    
    # Save to CSV (Google Sheets compatible)
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    logger.info(f"Classification results saved to {output_path}")
    
    return output_path


def export_summary_report(
    statistics: Dict[str, Any],
    unreliable_categories: Dict[str, List[str]],
    output_path: Optional[Path] = None
) -> Path:
    """
    Export summary report with statistics and unreliable categories.
    
    Args:
        statistics: Classification statistics
        unreliable_categories: Dictionary of unreliable categories
        output_path: Output file path
    
    Returns:
        Path to saved file
    """
    if output_path is None:
        output_path = get_output_path('task3_summary_report.json')
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'statistics': statistics,
        'unreliable_categories': unreliable_categories,
        'config': TASK3_CONFIG,
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Summary report saved to {output_path}")
    
    return output_path


# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """
    Main execution function for Task 3.
    """
    # Setup logging
    setup_logging('task3_spelling.log')
    
    logger.info("=" * 60)
    logger.info("Josh Talks AI/ML Internship - Task 3: Spelling Validation")
    logger.info("=" * 60)
    
    # ============================================
    # STEP 1: Load Unique Words Dataset
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Loading Unique Words Dataset")
    logger.info("=" * 60)
    
    try:
        df = pd.read_excel(get_data_path('Unique Words Data.xlsx'))
        logger.info(f"Loaded {len(df)} unique words from dataset")
    except FileNotFoundError:
        logger.error("Unique Words Data.xlsx not found. Using sample data for demonstration.")
        # Create sample data for demonstration
        sample_words = [
            'है', 'तो', 'और', 'में', 'का', 'की', 'के', 'को', 'से', 'पर',
            'प्रोजेक्ट', 'एरिया', 'टेंट', 'लाइट', 'मिस्टेक', 'कैम्प', 'गार्ड',
            'रोड', 'जंगल', 'फोन', 'कंप्यटूर', 'इंटरव्यू', 'जॉब', 'प्रॉब्लम',
            'खेतीबाडऱी', 'मोनता', 'बाडऱी', 'छै', 'मझु े', 'हमलोग',
            'अड़तीस', 'चौवन', 'पच्चीस', 'सौ', 'हज़ार', 'लाख',
            'दो-चार', 'छै सात', 'एक आधे',
            'राजस्थान', 'कोटा', 'प्रयागराज', 'कुंभ', 'मेला',
            'invalid१२३', 'test@word', 'broken##word',
        ]
        df = pd.DataFrame({'word': sample_words})
        logger.info(f"Created sample dataset with {len(df)} words")
    
    # Extract unique words
    unique_words = df['word'].dropna().unique().tolist()
    logger.info(f"Total unique words to classify: {len(unique_words)}")
    
    # ============================================
    # STEP 2: Initialize Classifier
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Initializing Spelling Classifier")
    logger.info("=" * 60)
    
    classifier = SpellingClassifier()
    
    # Load frequencies if available
    classifier.frequency_analyzer.load_frequencies_from_dataset(df)
    
    # ============================================
    # STEP 3: Classify All Words
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Classifying Words")
    logger.info("=" * 60)
    
    classifications = classifier.classify_batch(unique_words)
    
    # ============================================
    # STEP 4: Get Statistics
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Generating Statistics")
    logger.info("=" * 60)
    
    statistics = classifier.get_statistics()
    
    logger.info(f"Total words: {statistics.get('total_words', 0)}")
    logger.info(f"Correct: {statistics.get('correct_count', 0)} ({statistics.get('correct_percentage', 0):.1f}%)")
    logger.info(f"Incorrect: {statistics.get('incorrect_count', 0)} ({statistics.get('incorrect_percentage', 0):.1f}%)")
    logger.info(f"Uncertain: {statistics.get('uncertain_count', 0)}")
    logger.info(f"High confidence: {statistics.get('high_confidence_count', 0)}")
    logger.info(f"Medium confidence: {statistics.get('medium_confidence_count', 0)}")
    logger.info(f"Low confidence: {statistics.get('low_confidence_count', 0)}")
    
    # ============================================
    # STEP 5: Identify Unreliable Categories
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Identifying Unreliable Categories")
    logger.info("=" * 60)
    
    unreliable_categories = classifier.identify_unreliable_categories()
    
    for category, examples in unreliable_categories.items():
        logger.info(f"Category '{category}': {len(examples)} examples")
        if examples:
            logger.info(f"  Examples: {examples[:5]}")
    
    # ============================================
    # STEP 6: Sample Low-Confidence Words for Review
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Sampling Low-Confidence Words for Manual Review")
    logger.info("=" * 60)
    
    low_conf_sample = classifier.get_low_confidence_words(n=50)
    logger.info(f"Sampled {len(low_conf_sample)} low-confidence words for review")
    
    # Analyze sample (simulated - would need human annotation in production)
    simulated_accuracy = 0.75  # Simulated system accuracy on low-confidence words
    correct_predictions = int(len(low_conf_sample) * simulated_accuracy)
    incorrect_predictions = len(low_conf_sample) - correct_predictions
    
    logger.info(f"Simulated review results (would need human annotation):")
    logger.info(f"  System correct: ~{correct_predictions} ({simulated_accuracy*100:.0f}%)")
    logger.info(f"  System incorrect: ~{incorrect_predictions} ({(1-simulated_accuracy)*100:.0f}%)")
    
    # ============================================
    # STEP 7: Export Results
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 7: Exporting Results")
    logger.info("=" * 60)
    
    # Export classification results (Google Sheet format)
    results_path = export_classification_results(classifications)
    
    # Export summary report
    summary_path = export_summary_report(statistics, unreliable_categories)
    
    # Export low-confidence sample for review
    sample_data = []
    for c in low_conf_sample:
        sample_data.append({
            'word': c.word,
            'classification': c.classification,
            'confidence': c.confidence,
            'reason': c.reason,
            'category': c.category,
        })
    
    sample_path = get_output_path('task3_low_confidence_sample.csv')
    pd.DataFrame(sample_data).to_csv(sample_path, index=False, encoding='utf-8')
    logger.info(f"Low-confidence sample saved to {sample_path}")
    
    # ============================================
    # SUMMARY
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("TASK 3 COMPLETED - SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total unique words: {len(unique_words)}")
    logger.info(f"Correct spelling: ~{statistics.get('correct_count', 0)}")
    logger.info(f"Incorrect spelling: ~{statistics.get('incorrect_count', 0)}")
    logger.info(f"Uncertain (needs review): {statistics.get('uncertain_count', 0)}")
    logger.info(f"Unreliable categories identified: {len(unreliable_categories)}")
    logger.info(f"Results exported to: {results_path}")
    logger.info(f"Summary report: {summary_path}")
    logger.info(f"Low-confidence sample: {sample_path}")
    logger.info("=" * 60)
    logger.info("To launch Gradio review interface, run: python task3_spelling.py --review")
    logger.info("=" * 60)
    
    return {
        'total_words': len(unique_words),
        'statistics': statistics,
        'unreliable_categories': unreliable_categories,
        'results_path': str(results_path),
        'summary_path': str(summary_path),
        'sample_path': str(sample_path),
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Task 3: Spelling Validation Pipeline")
    parser.add_argument('--review', action='store_true', help='Launch Gradio review interface')
    parser.add_argument('--samples', type=int, default=50, help='Number of low-confidence words to sample')
    
    args = parser.parse_args()
    
    try:
        if args.review:
            # Launch Gradio review interface only
            classifier = SpellingClassifier()
            
            # Load and classify words first
            try:
                df = pd.read_excel(get_data_path('Unique Words Data.xlsx'))
                unique_words = df['word'].dropna().unique().tolist()[:1000]  # Limit for demo
                classifier.classify_batch(unique_words)
            except FileNotFoundError:
                logger.warning("Dataset not found. Launching with empty classifier.")
            
            create_gradio_review_interface(classifier)
        else:
            # Run full pipeline
            results = main()
            print("\n✅ Task 3 completed successfully!")
            print(f"Results: {results}")
    except Exception as e:
        logger.error(f"Task 3 failed: {e}")
        print(f"\n❌ Task 3 failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)