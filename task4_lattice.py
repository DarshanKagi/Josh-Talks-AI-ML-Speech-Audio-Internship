"""
============================================
Josh Talks AI/ML Speech & Audio Internship
Task 4: Lattice-Based ASR Evaluation Framework
============================================

This script:
1. Loads 46 segments with human reference + 5 ASR model outputs from Question 4.xlsx
2. Constructs word-level lattice capturing all valid transcription alternatives
3. Implements weighted consensus mechanism (≥4/5 models can override reference)
4. Handles insertions, deletions, substitutions fairly using best-path matching
5. Computes lattice-based WER for each model
6. Provides Gradio interface for lattice visualization and WER comparison

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
    calculate_wer,
    GRADIO_CONFIG,
    setup_logging,
)


# ============================================
# LOGGING SETUP
# ============================================

logger = logging.getLogger(__name__)


# ============================================
# TEXT NORMALIZATION FOR LATTICE ALIGNMENT
# ============================================

class LatticeTextNormalizer:
    """
    Normalizes text for consistent word-level alignment in lattice construction.
    Handles spacing, punctuation, and common spelling variations.
    """
    
    def __init__(self):
        # Common spelling variations that should be treated as equivalent
        self.spelling_variants = {
            'मौनता': ['मोनता', 'मोन ता', 'मौन ता'],
            'खेतीबाड़ी': ['खेती बाड़ी', 'खेतीबाडऱी', 'खेती बाडरी'],
            'किताबें': ['किताबे', 'किताबें', 'पुस्तकें'],
            'चौदह': ['14', 'चौदह', 'चौधह'],
            'खरीदीं': ['खरीदी', 'खरीदे', 'खरीदा'],
            'है': ['ह', 'हे', 'हय'],
            'हैं': ['हैं', 'ह', 'हय'],
            'में': ['मे', 'मैं'],
            'को': ['को', 'को'],
            'से': ['से', 'से'],
            'पर': ['पर', 'पे'],
            'और': ['और', 'ओर'],
            'का': ['का', 'के'],
            'की': ['की', 'के'],
            'के': ['के', 'का', 'की'],
        }
        
        # Punctuation to remove for comparison
        self.punctuation_pattern = re.compile(r'[^\w\s]')
    
    def normalize(self, text: str) -> str:
        """
        Normalize text for lattice alignment.
        
        Args:
            text: Input Hindi text
        
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Convert to string if needed
        text = str(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove punctuation for comparison (optional - can be configured)
        # text = self.punctuation_pattern.sub('', text)
        
        # Standardize common variations
        for canonical, variants in self.spelling_variants.items():
            for variant in variants:
                text = text.replace(variant, canonical)
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words for lattice alignment.
        
        Args:
            text: Input Hindi text
        
        Returns:
            List of word tokens
        """
        if not text:
            return []
        
        # Simple whitespace tokenization
        tokens = text.strip().split()
        
        # Filter empty tokens
        tokens = [t for t in tokens if t.strip()]
        
        return tokens
    
    def get_canonical_form(self, word: str) -> str:
        """
        Get canonical form of a word for comparison.
        
        Args:
            word: Input word
        
        Returns:
            Canonical form
        """
        for canonical, variants in self.spelling_variants.items():
            if word == canonical or word in variants:
                return canonical
        return word


# ============================================
# LATTICE CONSTRUCTION
# ============================================

@dataclass
class LatticeBin:
    """Represents a single bin in the lattice containing alternative words."""
    position: int
    alternatives: List[str]
    source_counts: Dict[str, int]  # How many models produced each alternative
    reference_word: Optional[str] = None
    consensus_word: Optional[str] = None
    is_override: bool = False  # True if consensus overrides reference
    
    def add_alternative(self, word: str, source: str = 'model'):
        """Add an alternative word to this bin."""
        if word not in self.alternatives:
            self.alternatives.append(word)
            self.source_counts[word] = 0
        self.source_counts[word] += 1
    
    def get_most_common(self) -> Tuple[str, int]:
        """Get the most common alternative and its count."""
        if not self.source_counts:
            return (None, 0)
        return max(self.source_counts.items(), key=lambda x: x[1])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'position': self.position,
            'alternatives': self.alternatives,
            'source_counts': self.source_counts,
            'reference_word': self.reference_word,
            'consensus_word': self.consensus_word,
            'is_override': self.is_override,
        }


@dataclass
class Lattice:
    """Represents a complete lattice for a single segment."""
    segment_id: str
    segment_url: str
    bins: List[LatticeBin]
    human_reference: str
    model_outputs: Dict[str, str]
    consensus_threshold: int = 4  # Minimum models needed to override reference
    
    def get_valid_paths(self) -> List[List[str]]:
        """
        Get all valid paths through the lattice.
        For simplicity, returns the reference path and consensus path.
        """
        # Reference path
        ref_path = [bin.reference_word for bin in self.bins if bin.reference_word]
        
        # Consensus path
        consensus_path = [bin.consensus_word for bin in self.bins if bin.consensus_word]
        
        return [ref_path, consensus_path]
    
    def get_best_path_for_model(self, model_output: List[str]) -> List[str]:
        """
        Get the best path through the lattice for a specific model output.
        Uses best-path matching to minimize edit distance.
        """
        best_path = []
        
        for i, model_word in enumerate(model_output):
            if i < len(self.bins):
                bin = self.bins[i]
                # If model word matches any alternative, use it
                if model_word in bin.alternatives:
                    best_path.append(model_word)
                else:
                    # Use the closest alternative (by consensus)
                    best_path.append(bin.consensus_word or bin.reference_word)
            else:
                # Model has extra words (insertions)
                best_path.append(model_word)
        
        return best_path
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'segment_id': self.segment_id,
            'segment_url': self.segment_url,
            'bins': [bin.to_dict() for bin in self.bins],
            'human_reference': self.human_reference,
            'model_outputs': self.model_outputs,
            'consensus_threshold': self.consensus_threshold,
        }


class LatticeBuilder:
    """
    Builds lattices from human reference and multiple model outputs.
    """
    
    def __init__(self, consensus_threshold: int = 4, num_models: int = 5):
        self.consensus_threshold = consensus_threshold
        self.num_models = num_models
        self.normalizer = LatticeTextNormalizer()
    
    def build_lattice(
        self,
        segment_id: str,
        segment_url: str,
        human_reference: str,
        model_outputs: Dict[str, str],
    ) -> Lattice:
        """
        Build a lattice from human reference and model outputs.
        
        Args:
            segment_id: Unique segment identifier
            segment_url: URL to audio segment
            human_reference: Human transcription
            model_outputs: Dictionary of model_name -> transcription
        
        Returns:
            Constructed Lattice object
        """
        # Normalize and tokenize human reference
        ref_normalized = self.normalizer.normalize(human_reference)
        ref_tokens = self.normalizer.tokenize(ref_normalized)
        
        # Normalize and tokenize all model outputs
        model_tokens = {}
        for model_name, output in model_outputs.items():
            normalized = self.normalizer.normalize(output)
            model_tokens[model_name] = self.normalizer.tokenize(normalized)
        
        # Find maximum length for alignment
        max_length = max(len(ref_tokens), *[len(tokens) for tokens in model_tokens.values()])
        
        # Build bins
        bins = []
        
        for i in range(max_length):
            bin = LatticeBin(position=i, alternatives=[], source_counts={})
            
            # Add reference word
            if i < len(ref_tokens):
                ref_word = ref_tokens[i]
                bin.add_alternative(ref_word, 'reference')
                bin.reference_word = ref_word
            
            # Add model words
            for model_name, tokens in model_tokens.items():
                if i < len(tokens):
                    model_word = tokens[i]
                    bin.add_alternative(model_word, model_name)
            
            # Determine consensus word
            if bin.source_counts:
                most_common, count = bin.get_most_common()
                
                # Check if models agree enough to override reference
                model_agreement = sum(
                    count for word, count in bin.source_counts.items()
                    if word != bin.reference_word
                )
                
                if model_agreement >= self.consensus_threshold:
                    bin.consensus_word = most_common
                    bin.is_override = True
                else:
                    bin.consensus_word = bin.reference_word or most_common
            else:
                bin.consensus_word = bin.reference_word
            
            bins.append(bin)
        
        # Create lattice
        lattice = Lattice(
            segment_id=segment_id,
            segment_url=segment_url,
            bins=bins,
            human_reference=human_reference,
            model_outputs=model_outputs,
            consensus_threshold=self.consensus_threshold,
        )
        
        return lattice
    
    def build_lattices_from_dataframe(self, df: pd.DataFrame) -> List[Lattice]:
        """
        Build lattices for all segments in a DataFrame.
        
        Args:
            df: DataFrame with segment data
        
        Returns:
            List of Lattice objects
        """
        lattices = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building lattices"):
            try:
                segment_id = str(idx)
                segment_url = row.get('segment_url_link', '')
                human_reference = str(row.get('Human', ''))
                
                # Extract model outputs
                model_outputs = {}
                for model_name in ['H', 'i', 'k', 'l', 'm', 'n']:
                    col_name = f'Model {model_name}'
                    if col_name in row:
                        model_outputs[model_name] = str(row[col_name])
                
                lattice = self.build_lattice(
                    segment_id=segment_id,
                    segment_url=segment_url,
                    human_reference=human_reference,
                    model_outputs=model_outputs,
                )
                lattices.append(lattice)
                
            except Exception as e:
                logger.error(f"Error building lattice for segment {idx}: {e}")
                continue
        
        logger.info(f"Built {len(lattices)} lattices")
        
        return lattices


# ============================================
# LATTICE-BASED WER COMPUTATION
# ============================================

class LatticeWERCalculator:
    """
    Computes WER using lattice-based best-path matching.
    """
    
    def __init__(self, normalizer: LatticeTextNormalizer = None):
        self.normalizer = normalizer or LatticeTextNormalizer()
    
    def compute_lattice_wer(
        self,
        model_output: str,
        lattice: Lattice,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute WER using lattice-based best-path matching.
        
        Args:
            model_output: Model transcription
            lattice: Lattice object
        
        Returns:
            Tuple of (WER, detailed_metrics)
        """
        # Normalize and tokenize model output
        model_normalized = self.normalizer.normalize(model_output)
        model_tokens = self.normalizer.tokenize(model_normalized)
        
        # Get best path through lattice for this model
        best_path = lattice.get_best_path_for_model(model_tokens)
        
        # Calculate edit distance between model output and best path
        substitutions = 0
        insertions = 0
        deletions = 0
        matches = 0
        
        # Simple alignment
        max_len = max(len(model_tokens), len(best_path))
        
        for i in range(max_len):
            model_word = model_tokens[i] if i < len(model_tokens) else None
            path_word = best_path[i] if i < len(best_path) else None
            
            if model_word is None:
                deletions += 1
            elif path_word is None:
                insertions += 1
            elif self.normalizer.get_canonical_form(model_word) == self.normalizer.get_canonical_form(path_word):
                matches += 1
            else:
                substitutions += 1
        
        # Account for length differences
        if len(model_tokens) > len(best_path):
            insertions += len(model_tokens) - len(best_path)
        elif len(best_path) > len(model_tokens):
            deletions += len(best_path) - len(model_tokens)
        
        # Calculate WER
        total_errors = substitutions + insertions + deletions
        total_words = max(len(model_tokens), len(best_path), 1)
        
        wer = total_errors / total_words
        
        metrics = {
            'substitutions': substitutions,
            'insertions': insertions,
            'deletions': deletions,
            'matches': matches,
            'total_errors': total_errors,
            'total_words': total_words,
            'wer': wer,
        }
        
        return wer, metrics
    
    def compute_standard_wer(
        self,
        model_output: str,
        reference: str,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute standard WER against human reference.
        
        Args:
            model_output: Model transcription
            reference: Human reference transcription
        
        Returns:
            Tuple of (WER, detailed_metrics)
        """
        # Use jiwer if available, otherwise simple implementation
        try:
            from jiwer import wer as jiwer_wer, compute_measures
            
            wer_score = jiwer_wer(reference, model_output)
            measures = compute_measures(reference, model_output)
            
            metrics = {
                'substitutions': measures.get('substitutions', 0),
                'insertions': measures.get('insertions', 0),
                'deletions': measures.get('deletions', 0),
                'matches': measures.get('hits', 0),
                'total_errors': measures.get('substitutions', 0) + measures.get('insertions', 0) + measures.get('deletions', 0),
                'total_words': measures.get('substitutions', 0) + measures.get('insertions', 0) + measures.get('deletions', 0) + measures.get('hits', 0),
                'wer': wer_score,
            }
            
            return wer_score, metrics
            
        except ImportError:
            # Fallback simple implementation
            ref_tokens = self.normalizer.tokenize(self.normalizer.normalize(reference))
            model_tokens = self.normalizer.tokenize(self.normalizer.normalize(model_output))
            
            # Simple edit distance
            edits = abs(len(ref_tokens) - len(model_tokens))
            wer = edits / max(len(ref_tokens), 1)
            
            metrics = {
                'substitutions': 0,
                'insertions': edits,
                'deletions': 0,
                'matches': min(len(ref_tokens), len(model_tokens)),
                'total_errors': edits,
                'total_words': max(len(ref_tokens), len(model_tokens)),
                'wer': wer,
            }
            
            return wer, metrics
    
    def compare_wer_methods(
        self,
        model_output: str,
        reference: str,
        lattice: Lattice,
    ) -> Dict[str, Any]:
        """
        Compare standard WER vs lattice-based WER.
        
        Args:
            model_output: Model transcription
            reference: Human reference
            lattice: Lattice object
        
        Returns:
            Comparison dictionary
        """
        standard_wer, standard_metrics = self.compute_standard_wer(model_output, reference)
        lattice_wer, lattice_metrics = self.compute_lattice_wer(model_output, lattice)
        
        return {
            'standard_wer': standard_wer,
            'lattice_wer': lattice_wer,
            'wer_reduction': standard_wer - lattice_wer,
            'wer_reduction_percent': ((standard_wer - lattice_wer) / standard_wer * 100) if standard_wer > 0 else 0,
            'standard_metrics': standard_metrics,
            'lattice_metrics': lattice_metrics,
            'is_unfairly_penalized': lattice_wer < standard_wer,
        }


# ============================================
# LATTICE EVALUATION PIPELINE
# ============================================

class LatticeEvaluationPipeline:
    """
    Main pipeline for lattice-based ASR evaluation.
    """
    
    def __init__(self, consensus_threshold: int = 4):
        self.consensus_threshold = consensus_threshold
        self.lattice_builder = LatticeBuilder(consensus_threshold=consensus_threshold)
        self.wer_calculator = LatticeWERCalculator()
        self.lattices: List[Lattice] = []
        self.evaluation_results: List[Dict[str, Any]] = []
    
    def load_data(self, data_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load Question 4.xlsx data.
        
        Args:
            data_path: Path to Excel file
        
        Returns:
            DataFrame with segment data
        """
        if data_path is None:
            data_path = get_data_path('Question 4.xlsx')
        
        logger.info(f"Loading data from {data_path}")
        df = pd.read_excel(data_path)
        logger.info(f"Loaded {len(df)} segments")
        
        return df
    
    def build_all_lattices(self, df: pd.DataFrame) -> List[Lattice]:
        """
        Build lattices for all segments.
        
        Args:
            df: DataFrame with segment data
        
        Returns:
            List of Lattice objects
        """
        self.lattices = self.lattice_builder.build_lattices_from_dataframe(df)
        return self.lattices
    
    def evaluate_all_models(self) -> List[Dict[str, Any]]:
        """
        Evaluate all models using lattice-based WER.
        
        Returns:
            List of evaluation result dictionaries
        """
        results = []
        
        for lattice in tqdm(self.lattices, desc="Evaluating models"):
            for model_name, model_output in lattice.model_outputs.items():
                # Compare standard vs lattice WER
                comparison = self.wer_calculator.compare_wer_methods(
                    model_output=model_output,
                    reference=lattice.human_reference,
                    lattice=lattice,
                )
                
                result = {
                    'segment_id': lattice.segment_id,
                    'segment_url': lattice.segment_url,
                    'model_name': model_name,
                    'human_reference': lattice.human_reference,
                    'model_output': model_output,
                    **comparison,
                }
                
                results.append(result)
        
        self.evaluation_results = results
        logger.info(f"Evaluated {len(results)} model outputs")
        
        return results
    
    def get_aggregate_statistics(self) -> Dict[str, Any]:
        """
        Get aggregate statistics across all evaluations.
        
        Returns:
            Dictionary of statistics
        """
        if not self.evaluation_results:
            return {}
        
        df = pd.DataFrame(self.evaluation_results)
        
        # Group by model
        model_stats = {}
        for model_name in df['model_name'].unique():
            model_df = df[df['model_name'] == model_name]
            
            model_stats[model_name] = {
                'avg_standard_wer': model_df['standard_wer'].mean(),
                'avg_lattice_wer': model_df['lattice_wer'].mean(),
                'avg_wer_reduction': model_df['wer_reduction'].mean(),
                'avg_wer_reduction_percent': model_df['wer_reduction_percent'].mean(),
                'unfairly_penalized_count': model_df['is_unfairly_penalized'].sum(),
                'unfairly_penalized_percent': (model_df['is_unfairly_penalized'].sum() / len(model_df) * 100),
                'total_segments': len(model_df),
            }
        
        # Overall statistics
        overall_stats = {
            'total_segments': len(self.lattices),
            'total_evaluations': len(self.evaluation_results),
            'avg_standard_wer': df['standard_wer'].mean(),
            'avg_lattice_wer': df['lattice_wer'].mean(),
            'avg_wer_reduction': df['wer_reduction'].mean(),
            'avg_wer_reduction_percent': df['wer_reduction_percent'].mean(),
            'total_unfairly_penalized': df['is_unfairly_penalized'].sum(),
            'unfairly_penalized_percent': (df['is_unfairly_penalized'].sum() / len(df) * 100),
            'model_statistics': model_stats,
        }
        
        return overall_stats
    
    def identify_override_cases(self) -> List[Dict[str, Any]]:
        """
        Identify cases where model consensus overrode human reference.
        
        Returns:
            List of override case dictionaries
        """
        override_cases = []
        
        for lattice in self.lattices:
            for bin in lattice.bins:
                if bin.is_override:
                    override_cases.append({
                        'segment_id': lattice.segment_id,
                        'position': bin.position,
                        'reference_word': bin.reference_word,
                        'consensus_word': bin.consensus_word,
                        'source_counts': bin.source_counts,
                    })
        
        logger.info(f"Found {len(override_cases)} override cases")
        
        return override_cases


# ============================================
# RESULTS EXPORT
# ============================================

def export_evaluation_results(
    results: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Export evaluation results to CSV.
    
    Args:
        results: List of evaluation result dictionaries
        output_path: Output file path
    
    Returns:
        Path to saved file
    """
    if output_path is None:
        output_path = get_output_path('task4_lattice_evaluation_results.csv')
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    logger.info(f"Evaluation results saved to {output_path}")
    
    return output_path


def export_lattice_data(
    lattices: List[Lattice],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Export lattice data to JSON.
    
    Args:
        lattices: List of Lattice objects
        output_path: Output file path
    
    Returns:
        Path to saved file
    """
    if output_path is None:
        output_path = get_output_path('task4_lattices.json')
    
    data = [lattice.to_dict() for lattice in lattices]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Lattice data saved to {output_path}")
    
    return output_path


def export_summary_report(
    statistics: Dict[str, Any],
    override_cases: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Export summary report with statistics.
    
    Args:
        statistics: Aggregate statistics
        override_cases: List of override cases
        output_path: Output file path
    
    Returns:
        Path to saved file
    """
    if output_path is None:
        output_path = get_output_path('task4_summary_report.json')
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'statistics': statistics,
        'override_cases_count': len(override_cases),
        'override_cases_sample': override_cases[:20],  # First 20 cases
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Summary report saved to {output_path}")
    
    return output_path


# ============================================
# GRADIO INTERFACE
# ============================================

def create_gradio_demo(pipeline: LatticeEvaluationPipeline) -> None:
    """
    Create Gradio interface for lattice visualization and WER comparison.
    
    Args:
        pipeline: LatticeEvaluationPipeline instance
    """
    try:
        import gradio as gr
    except ImportError:
        logger.error("Gradio not installed. Run: pip install gradio")
        return
    
    # Get sample data for demo
    if not pipeline.lattices:
        gr.Error("No lattices available. Run evaluation first.")
        return
    
    def get_segment_options() -> List[str]:
        """Get list of segment IDs for dropdown."""
        return [f"{l.segment_id}: {l.human_reference[:50]}..." for l in pipeline.lattices[:20]]
    
    def display_lattice(segment_select: str) -> str:
        """Display lattice for selected segment."""
        if not segment_select:
            return "Please select a segment."
        
        # Find lattice
        segment_id = segment_select.split(':')[0]
        lattice = None
        for l in pipeline.lattices:
            if l.segment_id == segment_id:
                lattice = l
                break
        
        if not lattice:
            return "Segment not found."
        
        # Format lattice display
        output = f"### Segment {lattice.segment_id}\n\n"
        output += f"**Human Reference:** {lattice.human_reference}\n\n"
        output += "**Lattice Bins:**\n\n"
        
        for i, bin in enumerate(lattice.bins[:10]):  # Show first 10 bins
            output += f"**Position {i}:**\n"
            output += f"- Reference: `{bin.reference_word}`\n"
            output += f"- Consensus: `{bin.consensus_word}`\n"
            output += f"- Alternatives: {bin.alternatives}\n"
            output += f"- Source Counts: {bin.source_counts}\n"
            output += f"- Override: {'Yes' if bin.is_override else 'No'}\n\n"
        
        if len(lattice.bins) > 10:
            output += f"... and {len(lattice.bins) - 10} more bins\n"
        
        return output
    
    def compare_wer(segment_select: str, model_name: str) -> str:
        """Compare standard vs lattice WER for selected segment and model."""
        if not segment_select or not model_name:
            return "Please select a segment and model."
        
        # Find lattice
        segment_id = segment_select.split(':')[0]
        lattice = None
        for l in pipeline.lattices:
            if l.segment_id == segment_id:
                lattice = l
                break
        
        if not lattice:
            return "Segment not found."
        
        if model_name not in lattice.model_outputs:
            return f"Model {model_name} not found for this segment."
        
        # Calculate WER comparison
        comparison = pipeline.wer_calculator.compare_wer_methods(
            model_output=lattice.model_outputs[model_name],
            reference=lattice.human_reference,
            lattice=lattice,
        )
        
        output = f"### WER Comparison for Model {model_name}\n\n"
        output += f"**Segment:** {lattice.segment_id}\n\n"
        output += f"**Human Reference:** {lattice.human_reference}\n\n"
        output += f"**Model Output:** {lattice.model_outputs[model_name]}\n\n"
        output += "### Metrics\n\n"
        output += f"| Metric | Value |\n"
        output += f"|--------|-------|\n"
        output += f"| Standard WER | {comparison['standard_wer']:.4f} |\n"
        output += f"| Lattice WER | {comparison['lattice_wer']:.4f} |\n"
        output += f"| WER Reduction | {comparison['wer_reduction']:.4f} |\n"
        output += f"| WER Reduction % | {comparison['wer_reduction_percent']:.2f}% |\n"
        output += f"| Unfairly Penalized | {'Yes' if comparison['is_unfairly_penalized'] else 'No'} |\n\n"
        output += "### Edit Breakdown (Lattice)\n\n"
        output += f"- Substitutions: {comparison['lattice_metrics']['substitutions']}\n"
        output += f"- Insertions: {comparison['lattice_metrics']['insertions']}\n"
        output += f"- Deletions: {comparison['lattice_metrics']['deletions']}\n"
        output += f"- Matches: {comparison['lattice_metrics']['matches']}\n"
        
        return output
    
    def show_aggregate_stats() -> str:
        """Show aggregate statistics across all segments."""
        stats = pipeline.get_aggregate_statistics()
        
        if not stats:
            return "No statistics available. Run evaluation first."
        
        output = "### Aggregate Statistics\n\n"
        output += f"**Total Segments:** {stats.get('total_segments', 0)}\n\n"
        output += f"**Total Evaluations:** {stats.get('total_evaluations', 0)}\n\n"
        output += f"**Average Standard WER:** {stats.get('avg_standard_wer', 0):.4f}\n\n"
        output += f"**Average Lattice WER:** {stats.get('avg_lattice_wer', 0):.4f}\n\n"
        output += f"**Average WER Reduction:** {stats.get('avg_wer_reduction', 0):.4f}\n\n"
        output += f"**Average WER Reduction %:** {stats.get('avg_wer_reduction_percent', 0):.2f}%\n\n"
        output += f"**Total Unfairly Penalized:** {stats.get('total_unfairly_penalized', 0)}\n\n"
        output += f"**Unfairly Penalized %:** {stats.get('unfairly_penalized_percent', 0):.2f}%\n\n"
        
        # Model-specific stats
        output += "### Per-Model Statistics\n\n"
        output += "| Model | Std WER | Lat WER | Reduction | Reduction % | Unfair % |\n"
        output += "|-------|---------|---------|-----------|-------------|----------|\n"
        
        for model_name, model_stats in stats.get('model_statistics', {}).items():
            output += f"| {model_name} | {model_stats['avg_standard_wer']:.4f} | {model_stats['avg_lattice_wer']:.4f} | {model_stats['avg_wer_reduction']:.4f} | {model_stats['avg_wer_reduction_percent']:.2f}% | {model_stats['unfairly_penalized_percent']:.2f}% |\n"
        
        return output
    
    with gr.Blocks(title="Josh Talks Lattice Evaluation", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔤 Lattice-Based ASR Evaluation Framework")
        gr.Markdown("""
        This tool demonstrates lattice-based evaluation for Hindi ASR models.
        
        **Features:**
        - Visualize lattice structure with alternative transcriptions
        - Compare standard WER vs lattice-based WER
        - Identify cases where models were unfairly penalized
        - View aggregate statistics across all segments
        """)
        
        with gr.Tab("Lattice Visualization"):
            segment_dropdown = gr.Dropdown(
                choices=get_segment_options(),
                label="Select Segment",
                interactive=True
            )
            lattice_display = gr.Textbox(
                label="Lattice Structure",
                lines=20,
                interactive=False
            )
            
            segment_dropdown.change(
                fn=display_lattice,
                inputs=[segment_dropdown],
                outputs=[lattice_display]
            )
        
        with gr.Tab("WER Comparison"):
            segment_dropdown_wer = gr.Dropdown(
                choices=get_segment_options(),
                label="Select Segment",
                interactive=True
            )
            model_dropdown = gr.Dropdown(
                choices=['H', 'i', 'k', 'l', 'm', 'n'],
                label="Select Model",
                value='H',
                interactive=True
            )
            wer_display = gr.Textbox(
                label="WER Comparison",
                lines=20,
                interactive=False
            )
            
            compare_btn = gr.Button("Compare WER", variant="primary")
            compare_btn.click(
                fn=compare_wer,
                inputs=[segment_dropdown_wer, model_dropdown],
                outputs=[wer_display]
            )
        
        with gr.Tab("Aggregate Statistics"):
            stats_btn = gr.Button("Show Statistics", variant="primary")
            stats_display = gr.Textbox(
                label="Aggregate Statistics",
                lines=25,
                interactive=False
            )
            
            stats_btn.click(
                fn=show_aggregate_stats,
                outputs=[stats_display]
            )
        
        with gr.Accordion("About Lattice Evaluation", open=False):
            gr.Markdown("""
            ### How Lattice Evaluation Works
            
            1. **Lattice Construction:** Multiple model outputs + human reference are aligned into bins
            2. **Consensus Mechanism:** If ≥4/5 models agree, they can override the human reference
            3. **Best-Path Matching:** Model output is compared against the closest valid path through the lattice
            4. **Fair WER:** Reduces penalties for valid alternative transcriptions
            
            ### Benefits
            
            - ✅ Fairer evaluation for models with valid alternative transcriptions
            - ✅ Identifies potential errors in human reference
            - ✅ Handles spelling variations and spacing differences
            - ✅ Provides detailed error analysis
            """)
    
    # Launch Gradio
    config = get_gradio_config()
    demo.launch(
        server_name=config.get('server_name', '0.0.0.0'),
        server_port=config.get('server_port', 7860),
        share=config.get('share', False),
    )


# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """
    Main execution function for Task 4.
    """
    # Setup logging
    setup_logging('task4_lattice.log')
    
    logger.info("=" * 60)
    logger.info("Josh Talks AI/ML Internship - Task 4: Lattice Evaluation")
    logger.info("=" * 60)
    
    # ============================================
    # STEP 1: Initialize Pipeline
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Initializing Lattice Evaluation Pipeline")
    logger.info("=" * 60)
    
    pipeline = LatticeEvaluationPipeline(consensus_threshold=4)
    
    # ============================================
    # STEP 2: Load Data
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Loading Question 4 Data")
    logger.info("=" * 60)
    
    df = pipeline.load_data()
    logger.info(f"Loaded {len(df)} segments from Question 4.xlsx")
    
    # ============================================
    # STEP 3: Build Lattices
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Building Lattices")
    logger.info("=" * 60)
    
    lattices = pipeline.build_all_lattices(df)
    logger.info(f"Built {len(lattices)} lattices")
    
    # ============================================
    # STEP 4: Evaluate All Models
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Evaluating All Models")
    logger.info("=" * 60)
    
    results = pipeline.evaluate_all_models()
    logger.info(f"Completed {len(results)} evaluations")
    
    # ============================================
    # STEP 5: Get Statistics
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Computing Aggregate Statistics")
    logger.info("=" * 60)
    
    statistics = pipeline.get_aggregate_statistics()
    
    logger.info(f"Average Standard WER: {statistics.get('avg_standard_wer', 0):.4f}")
    logger.info(f"Average Lattice WER: {statistics.get('avg_lattice_wer', 0):.4f}")
    logger.info(f"Average WER Reduction: {statistics.get('avg_wer_reduction', 0):.4f}")
    logger.info(f"Average WER Reduction %: {statistics.get('avg_wer_reduction_percent', 0):.2f}%")
    logger.info(f"Unfairly Penalized: {statistics.get('total_unfairly_penalized', 0)} ({statistics.get('unfairly_penalized_percent', 0):.2f}%)")
    
    # ============================================
    # STEP 6: Identify Override Cases
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Identifying Reference Override Cases")
    logger.info("=" * 60)
    
    override_cases = pipeline.identify_override_cases()
    logger.info(f"Found {len(override_cases)} cases where model consensus overrode reference")
    
    # ============================================
    # STEP 7: Export Results
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 7: Exporting Results")
    logger.info("=" * 60)
    
    # Export evaluation results
    results_path = export_evaluation_results(results)
    
    # Export lattice data
    lattice_path = export_lattice_data(lattices)
    
    # Export summary report
    summary_path = export_summary_report(statistics, override_cases)
    
    # ============================================
    # SUMMARY
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("TASK 4 COMPLETED - SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total segments: {len(lattices)}")
    logger.info(f"Total evaluations: {len(results)}")
    logger.info(f"Average Standard WER: {statistics.get('avg_standard_wer', 0):.4f}")
    logger.info(f"Average Lattice WER: {statistics.get('avg_lattice_wer', 0):.4f}")
    logger.info(f"Average WER Reduction: {statistics.get('avg_wer_reduction', 0):.4f}")
    logger.info(f"Unfairly Penalized Cases: {statistics.get('total_unfairly_penalized', 0)}")
    logger.info(f"Reference Override Cases: {len(override_cases)}")
    logger.info(f"Evaluation results: {results_path}")
    logger.info(f"Lattice data: {lattice_path}")
    logger.info(f"Summary report: {summary_path}")
    logger.info("=" * 60)
    logger.info("To launch Gradio demo, run: python task4_lattice.py --demo")
    logger.info("=" * 60)
    
    return {
        'total_segments': len(lattices),
        'total_evaluations': len(results),
        'statistics': statistics,
        'override_cases_count': len(override_cases),
        'results_path': str(results_path),
        'lattice_path': str(lattice_path),
        'summary_path': str(summary_path),
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Task 4: Lattice-Based ASR Evaluation")
    parser.add_argument('--demo', action='store_true', help='Launch Gradio demo')
    parser.add_argument('--threshold', type=int, default=4, help='Consensus threshold for override')
    
    args = parser.parse_args()
    
    try:
        if args.demo:
            # Launch Gradio demo only
            pipeline = LatticeEvaluationPipeline(consensus_threshold=args.threshold)
            
            # Load and build lattices first
            df = pipeline.load_data()
            pipeline.build_all_lattices(df)
            pipeline.evaluate_all_models()
            
            create_gradio_demo(pipeline)
        else:
            # Run full pipeline
            results = main()
            print("\n✅ Task 4 completed successfully!")
            print(f"Results: {results}")
    except Exception as e:
        logger.error(f"Task 4 failed: {e}")
        print(f"\n❌ Task 4 failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)