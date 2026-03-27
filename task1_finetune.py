"""
============================================
Josh Talks AI/ML Speech & Audio Internship
Task 1: Whisper-Small Fine-Tuning on Hindi ASR
============================================

This script:
1. Loads and preprocesses the ~10-hour Hindi dataset from FT Data.xlsx
2. Corrects GCS URLs programmatically
3. Fetches audio and transcription data from Google Cloud Storage
4. Fine-tunes Whisper-small model on the Hindi dataset
5. Evaluates on FLEURS Hindi test set
6. Calculates and saves WER results

No hardcoded paths - all data access is dynamic via config.py
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import requests
import librosa
from tqdm import tqdm

# Hugging Face imports
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    WhisperFeatureExtractor,
    WhisperTokenizer,
)
from datasets import Dataset, DatasetDict, Audio
from jiwer import wer

# Local config import
from config import (
    get_project_root,
    get_data_path,
    get_output_path,
    get_model_path,
    correct_gcs_url,
    correct_gcs_urls_in_dataframe,
    load_ft_data,
    concatenate_transcription_segments,
    normalize_hindi_text,
    calculate_wer,
    save_results_to_excel,
    save_results_to_csv,
    TASK1_CONFIG,
    GRADIO_CONFIG,
    setup_logging,
)


# ============================================
# LOGGING SETUP
# ============================================

logger = logging.getLogger(__name__)


# ============================================
# DATA LOADING AND PREPROCESSING
# ============================================

@dataclass
class TranscriptionSample:
    """Represents a single audio-transcription pair."""
    recording_id: str
    user_id: str
    audio_path: str
    audio_array: np.ndarray
    sampling_rate: int
    text: str
    duration: float
    language: str = "hi"


class HindiASRDataset:
    """
    Handles loading and preprocessing of the Hindi ASR dataset.
    """
    
    def __init__(self, max_samples: Optional[int] = None):
        self.max_samples = max_samples
        self.samples: List[TranscriptionSample] = []
        self.failed_loads: List[Dict[str, Any]] = []
        
    def load_from_excel(self, excel_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load the FT Data.xlsx manifest file.
        
        Args:
            excel_path: Path to Excel file. If None, uses config to find it.
        
        Returns:
            DataFrame with corrected URLs
        """
        if excel_path is None:
            excel_path = get_data_path('FT Data.xlsx')
        
        logger.info(f"Loading dataset manifest from: {excel_path}")
        
        # Load Excel file
        df = pd.read_excel(excel_path)
        logger.info(f"Loaded {len(df)} recordings from manifest")
        
        # Correct GCS URLs
        df = correct_gcs_urls_in_dataframe(df)
        logger.info("Corrected GCS URLs in dataset")
        
        # Limit samples if specified
        if self.max_samples:
            df = df.head(self.max_samples)
            logger.info(f"Limited to {self.max_samples} samples")
        
        return df
    
    def fetch_audio_from_url(self, url: str, target_sr: int = 16000) -> Optional[Tuple[np.ndarray, int]]:
        """
        Fetch audio file from GCS URL and load into memory.
        
        Args:
            url: GCS URL to audio file
            target_sr: Target sampling rate (default 16000 for Whisper)
        
        Returns:
            Tuple of (audio_array, sampling_rate) or None if failed
        """
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            # Load audio from bytes
            audio_array, sr = librosa.load(
                io.BytesIO(response.content),
                sr=target_sr,
                mono=True
            )
            
            return audio_array, sr
            
        except Exception as e:
            logger.error(f"Failed to fetch audio from {url}: {e}")
            return None
    
    def fetch_transcription_from_url(self, url: str) -> Optional[str]:
        """
        Fetch and parse transcription JSON from GCS URL.
        Concatenates all segments into single transcript.
        
        Args:
            url: GCS URL to transcription JSON
        
        Returns:
            Concatenated transcription text or None if failed
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Concatenate all segment texts
            text = concatenate_transcription_segments(data)
            
            return text
            
        except Exception as e:
            logger.error(f"Failed to fetch transcription from {url}: {e}")
            return None
    
    def build_dataset(self, df: pd.DataFrame) -> Dataset:
        """
        Build Hugging Face Dataset from DataFrame.
        
        Args:
            df: DataFrame with corrected URLs
        
        Returns:
            Hugging Face Dataset ready for training
        """
        audio_data = []
        texts = []
        recording_ids = []
        durations = []
        failed_indices = []
        
        logger.info(f"Building dataset from {len(df)} recordings...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading audio & transcriptions"):
            try:
                # Fetch audio
                audio_result = self.fetch_audio_from_url(row['rec_url_gcp'])
                if audio_result is None:
                    failed_indices.append(idx)
                    self.failed_loads.append({
                        'recording_id': row['recording_id'],
                        'error': 'Audio fetch failed'
                    })
                    continue
                
                audio_array, sr = audio_result
                
                # Fetch transcription
                text = self.fetch_transcription_from_url(row['transcription_url_gcp'])
                if text is None or len(text.strip()) == 0:
                    failed_indices.append(idx)
                    self.failed_loads.append({
                        'recording_id': row['recording_id'],
                        'error': 'Transcription fetch failed'
                    })
                    continue
                
                # Add to dataset
                audio_data.append({
                    'array': audio_array,
                    'sampling_rate': sr
                })
                texts.append(text.strip())
                recording_ids.append(str(row['recording_id']))
                durations.append(row['duration'])
                
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                failed_indices.append(idx)
                self.failed_loads.append({
                    'recording_id': row.get('recording_id', 'unknown'),
                    'error': str(e)
                })
        
        logger.info(f"Successfully loaded {len(audio_data)} samples, {len(failed_indices)} failed")
        
        # Create Hugging Face Dataset
        dataset = Dataset.from_dict({
            'recording_id': recording_ids,
            'audio': audio_data,
            'text': texts,
            'duration': durations,
        })
        
        # Cast audio column to proper Audio type
        dataset = dataset.cast_column('audio', Audio(sampling_rate=16000))
        
        return dataset
    
    def save_failed_loads(self, output_path: Optional[Path] = None):
        """Save information about failed loads for debugging."""
        if output_path is None:
            output_path = get_output_path('task1_failed_loads.json')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.failed_loads, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved failed load information to {output_path}")


# ============================================
# DATA PREPROCESSING FOR WHISPER
# ============================================

class WhisperDataProcessor:
    """
    Handles data preprocessing specific to Whisper model.
    """
    
    def __init__(self, model_name: str = "openai/whisper-small", language: str = "hindi"):
        self.model_name = model_name
        self.language = language
        
        # Load processor (includes feature extractor and tokenizer)
        logger.info(f"Loading Whisper processor from {model_name}")
        self.processor = WhisperProcessor.from_pretrained(
            model_name,
            language=language,
            task="transcribe"
        )
        
        # Get forced decoder IDs for Hindi
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language,
            task="transcribe"
        )
        
        logger.info("Whisper processor loaded successfully")
    
    def prepare_dataset(self, dataset: Dataset, split: str = "train") -> Dataset:
        """
        Prepare dataset for Whisper training.
        
        Args:
            dataset: Raw dataset with audio and text
            split: Dataset split name
        
        Returns:
            Processed dataset ready for training
        """
        def preprocess_function(batch):
            # Get audio arrays
            audio = batch["audio"]
            
            # Compute log-Mel spectrogram features
            batch["input_features"] = self.processor(
                audio["array"],
                sampling_rate=audio["sampling_rate"],
                return_tensors="pt"
            ).input_features[0]
            
            # Encode target text
            batch["labels"] = self.processor(
                text=audio.get("text", batch.get("text", "")),
                return_tensors="pt"
            ).input_ids[0]
            
            return batch
        
        # Remove columns not needed
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names if col not in ["audio", "text"]]
        )
        
        # Apply preprocessing
        dataset = dataset.map(
            preprocess_function,
            remove_columns=dataset.column_names,
            num_proc=4,
            desc="Preprocessing dataset"
        )
        
        return dataset
    
    def data_collator(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Custom data collator for Whisper training.
        
        Args:
            features: List of processed feature dictionaries
        
        Returns:
            Batched tensors ready for model
        """
        # Get input features and labels
        input_features = [{"input_features": f["input_features"]} for f in features]
        labels = [{"input_ids": f["labels"]} for f in features]
        
        # Batch input features
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt"
        )
        
        # Pad labels
        labels_batch = self.processor.tokenizer.pad(
            labels,
            return_tensors="pt"
        )
        
        # Replace padding with -100 (ignored in loss calculation)
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100
        )
        
        # Add labels to batch
        batch["labels"] = labels
        
        return batch


# ============================================
# MODEL TRAINING
# ============================================

class WhisperFineTuner:
    """
    Handles Whisper model fine-tuning.
    """
    
    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        language: str = "hindi",
        output_dir: Optional[Path] = None,
    ):
        self.model_name = model_name
        self.language = language
        
        if output_dir is None:
            output_dir = get_model_path("whisper-hindi-ft")
        self.output_dir = output_dir
        
        # Load model
        logger.info(f"Loading Whisper model from {model_name}")
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.config.forced_decoder_ids = None  # Will be set in generation
        
        logger.info(f"Model loaded. Parameters: {self.model.num_parameters():,}")
    
    def create_training_args(self, config: Dict[str, Any]) -> Seq2SeqTrainingArguments:
        """
        Create training arguments from config.
        
        Args:
            config: Training configuration dictionary
        
        Returns:
            Seq2SeqTrainingArguments
        """
        return Seq2SeqTrainingArguments(
            output_dir=str(self.output_dir),
            per_device_train_batch_size=config.get('batch_size', 8),
            gradient_accumulation_steps=2,
            learning_rate=config.get('learning_rate', 1e-5),
            warmup_steps=config.get('warmup_steps', 500),
            num_train_epochs=config.get('num_epochs', 10),
            evaluation_strategy=config.get('evaluation_strategy', 'epoch'),
            save_strategy=config.get('save_strategy', 'epoch'),
            logging_steps=25,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),
            predict_with_generate=True,
            generation_max_length=225,
            save_total_limit=2,
            report_to="none",  # Disable wandb for now
        )
    
    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """
        Compute WER metric for evaluation.
        
        Args:
            eval_pred: Evaluation predictions and labels
        
        Returns:
            Dictionary with WER metric
        """
        predictions, labels = eval_pred
        
        # Decode predictions
        decoded_preds = self.processor.batch_decode(
            predictions,
            skip_special_tokens=True
        )
        
        # Replace -100 in labels
        labels = np.where(labels != -100, labels, self.processor.tokenizer.pad_token_id)
        decoded_labels = self.processor.batch_decode(
            labels,
            skip_special_tokens=True
        )
        
        # Compute WER
        wer_score = wer(decoded_labels, decoded_preds)
        
        return {"wer": wer_score}
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Seq2SeqTrainer:
        """
        Fine-tune the Whisper model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            config: Training configuration
        
        Returns:
            Trained Seq2SeqTrainer
        """
        if config is None:
            config = TASK1_CONFIG
        
        # Create training arguments
        training_args = self.create_training_args(config)
        
        # Create data processor
        data_processor = WhisperDataProcessor(self.model_name, self.language)
        
        # Prepare datasets
        train_dataset = data_processor.prepare_dataset(train_dataset, "train")
        if eval_dataset is not None:
            eval_dataset = data_processor.prepare_dataset(eval_dataset, "eval")
        
        # Create trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=data_processor.processor.feature_extractor,
            data_collator=data_processor.data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        # Start training
        logger.info("Starting fine-tuning...")
        train_result = trainer.train()
        
        # Save model
        trainer.save_model(str(self.output_dir))
        logger.info(f"Model saved to {self.output_dir}")
        
        # Save training metrics
        metrics = train_result.metrics
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        logger.info(f"Training completed. Metrics: {metrics}")
        
        return trainer


# ============================================
# MODEL EVALUATION
# ============================================

class WhisperEvaluator:
    """
    Handles model evaluation on FLEURS test set.
    """
    
    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        language: str = "hindi",
    ):
        self.model_name = model_name
        self.language = language
        
        # Load processor
        self.processor = WhisperProcessor.from_pretrained(
            model_name,
            language=language,
            task="transcribe"
        )
    
    def load_fleurs_dataset(self, split: str = "test", language: str = "hi_in") -> Dataset:
        """
        Load FLEURS Hindi test dataset.
        
        Args:
            split: Dataset split
            language: Language code
        
        Returns:
            FLEURS dataset
        """
        try:
            from datasets import load_dataset
            
            logger.info(f"Loading FLEURS {language} {split} set...")
            dataset = load_dataset(
                "google/fleurs",
                language,
                split=split
            )
            
            logger.info(f"Loaded {len(dataset)} samples from FLEURS")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load FLEURS dataset: {e}")
            # Return empty dataset as fallback
            return Dataset.from_dict({
                'audio': [],
                'transcription': []
            })
    
    def evaluate_model(
        self,
        model: WhisperForConditionalGeneration,
        dataset: Dataset,
        max_samples: Optional[int] = None,
    ) -> float:
        """
        Evaluate model on dataset and return WER.
        
        Args:
            model: Whisper model to evaluate
            dataset: Dataset with audio and transcription
            max_samples: Maximum samples to evaluate
        
        Returns:
            Word Error Rate (WER)
        """
        if len(dataset) == 0:
            logger.warning("Empty dataset, returning WER=1.0")
            return 1.0
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        references = []
        predictions = []
        
        logger.info(f"Evaluating on {len(dataset)} samples...")
        
        for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc="Evaluating"):
            try:
                # Get audio
                audio = item["audio"]
                audio_array = audio["array"]
                sampling_rate = audio["sampling_rate"]
                
                # Get reference transcription
                reference = item.get("transcription", item.get("text", ""))
                
                # Prepare input
                inputs = self.processor(
                    audio_array,
                    sampling_rate=sampling_rate,
                    return_tensors="pt",
                    language=self.language,
                    task="transcribe"
                )
                
                # Generate prediction
                with torch.no_grad():
                    pred_ids = model.generate(
                        inputs.input_features,
                        max_length=225,
                        language=self.language,
                        task="transcribe"
                    )
                
                # Decode prediction
                prediction = self.processor.batch_decode(
                    pred_ids,
                    skip_special_tokens=True
                )[0]
                
                references.append(reference.strip())
                predictions.append(prediction.strip())
                
            except Exception as e:
                logger.error(f"Error evaluating sample {idx}: {e}")
                continue
        
        # Calculate WER
        if len(references) == 0:
            return 1.0
        
        wer_score = wer(references, predictions)
        logger.info(f"Evaluation complete. WER: {wer_score:.4f}")
        
        return wer_score
    
    def evaluate_baseline_and_finetuned(
        self,
        finetuned_model_path: Optional[Path] = None,
        max_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate both baseline (pretrained) and fine-tuned models.
        
        Args:
            finetuned_model_path: Path to fine-tuned model
            max_samples: Maximum samples for evaluation
        
        Returns:
            Dictionary with WER for both models
        """
        # Load FLEURS dataset
        fleurs_dataset = self.load_fleurs_dataset()
        
        results = {}
        
        # Evaluate baseline (pretrained) model
        logger.info("Evaluating baseline (pretrained) Whisper-small...")
        baseline_model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
        baseline_wer = self.evaluate_model(
            baseline_model,
            fleurs_dataset,
            max_samples=max_samples
        )
        results['baseline_wer'] = baseline_wer
        logger.info(f"Baseline WER: {baseline_wer:.4f}")
        
        # Evaluate fine-tuned model
        if finetuned_model_path and finetuned_model_path.exists():
            logger.info(f"Evaluating fine-tuned model from {finetuned_model_path}...")
            finetuned_model = WhisperForConditionalGeneration.from_pretrained(
                str(finetuned_model_path)
            )
            finetuned_wer = self.evaluate_model(
                finetuned_model,
                fleurs_dataset,
                max_samples=max_samples
            )
            results['finetuned_wer'] = finetuned_wer
            logger.info(f"Fine-tuned WER: {finetuned_wer:.4f}")
        else:
            logger.warning("Fine-tuned model not found, skipping evaluation")
            results['finetuned_wer'] = None
        
        return results


# ============================================
# RESULTS AND REPORTING
# ============================================

def save_wer_results(
    baseline_wer: float,
    finetuned_wer: Optional[float],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Save WER results to Excel file matching FT Result.xlsx format.
    
    Args:
        baseline_wer: Baseline model WER
        finetuned_wer: Fine-tuned model WER
        output_path: Output file path
    
    Returns:
        Path to saved file
    """
    if output_path is None:
        output_path = get_output_path('FT_Result.xlsx')
    
    # Create DataFrame matching expected format
    data = {
        'Model': ['Whisper Small (Pretrained)', 'FT Whisper Small (yours)'],
        'Hindi': [baseline_wer, finetuned_wer if finetuned_wer else 'To be Filled']
    }
    
    df = pd.DataFrame(data)
    
    # Save to Excel
    df.to_excel(output_path, index=False, sheet_name='Sheet1')
    
    logger.info(f"WER results saved to {output_path}")
    
    return output_path


def create_training_report(
    train_metrics: Dict[str, Any],
    eval_metrics: Dict[str, float],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Create comprehensive training report.
    
    Args:
        train_metrics: Training metrics
        eval_metrics: Evaluation metrics (WER)
        output_path: Output file path
    
    Returns:
        Path to saved report
    """
    if output_path is None:
        output_path = get_output_path('task1_training_report.json')
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model_name': TASK1_CONFIG['model_name'],
        'language': TASK1_CONFIG['language'],
        'training_metrics': train_metrics,
        'evaluation_metrics': {
            'baseline_wer': eval_metrics.get('baseline_wer'),
            'finetuned_wer': eval_metrics.get('finetuned_wer'),
            'wer_improvement': (
                eval_metrics.get('baseline_wer', 0) - eval_metrics.get('finetuned_wer', 0)
                if eval_metrics.get('finetuned_wer') else None
            ),
        },
        'config': TASK1_CONFIG,
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Training report saved to {output_path}")
    
    return output_path


# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """
    Main execution function for Task 1.
    """
    # Setup logging
    setup_logging('task1_training.log')
    
    logger.info("=" * 60)
    logger.info("Josh Talks AI/ML Internship - Task 1: Whisper Fine-Tuning")
    logger.info("=" * 60)
    
    # Check for GPU
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("No GPU available. Training will be slower.")
    
    # ============================================
    # STEP 1: Load and Preprocess Dataset
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Loading and Preprocessing Dataset")
    logger.info("=" * 60)
    
    dataset_loader = HindiASRDataset()
    df = dataset_loader.load_from_excel()
    dataset = dataset_loader.build_dataset(df)
    dataset_loader.save_failed_loads()
    
    logger.info(f"Dataset size: {len(dataset)} samples")
    logger.info(f"Dataset columns: {dataset.column_names}")
    
    # Split into train/eval
    dataset_dict = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_dict['train']
    eval_dataset = dataset_dict['test']
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Eval samples: {len(eval_dataset)}")
    
    # ============================================
    # STEP 2: Fine-Tune Whisper Model
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Fine-Tuning Whisper Model")
    logger.info("=" * 60)
    
    finetuner = WhisperFineTuner(
        model_name=TASK1_CONFIG['model_name'],
        language=TASK1_CONFIG['language'],
    )
    
    trainer = finetuner.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=TASK1_CONFIG,
    )
    
    train_metrics = trainer.state.log_history
    
    # ============================================
    # STEP 3: Evaluate Models
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Evaluating Models on FLEURS Test Set")
    logger.info("=" * 60)
    
    evaluator = WhisperEvaluator(
        model_name=TASK1_CONFIG['model_name'],
        language=TASK1_CONFIG['language'],
    )
    
    eval_results = evaluator.evaluate_baseline_and_finetuned(
        finetuned_model_path=finetuner.output_dir,
        max_samples=100,  # Limit for faster evaluation
    )
    
    # ============================================
    # STEP 4: Save Results
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Saving Results")
    logger.info("=" * 60)
    
    # Save WER results
    wer_file = save_wer_results(
        baseline_wer=eval_results.get('baseline_wer', 0.83),
        finetuned_wer=eval_results.get('finetuned_wer'),
    )
    
    # Save training report
    report_file = create_training_report(
        train_metrics=train_metrics,
        eval_metrics=eval_results,
    )
    
    # ============================================
    # SUMMARY
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("TASK 1 COMPLETED - SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    logger.info(f"Train/Eval split: {len(train_dataset)}/{len(eval_dataset)}")
    logger.info(f"Baseline WER: {eval_results.get('baseline_wer', 'N/A')}")
    logger.info(f"Fine-tuned WER: {eval_results.get('finetuned_wer', 'N/A')}")
    logger.info(f"WER Results saved to: {wer_file}")
    logger.info(f"Training report saved to: {report_file}")
    logger.info(f"Model saved to: {finetuner.output_dir}")
    logger.info("=" * 60)
    
    return {
        'dataset_size': len(dataset),
        'baseline_wer': eval_results.get('baseline_wer'),
        'finetuned_wer': eval_results.get('finetuned_wer'),
        'model_path': str(finetuner.output_dir),
        'wer_file': str(wer_file),
        'report_file': str(report_file),
    }


if __name__ == "__main__":
    try:
        results = main()
        print("\n✅ Task 1 completed successfully!")
        print(f"Results: {results}")
    except Exception as e:
        logger.error(f"Task 1 failed: {e}")
        print(f"\n❌ Task 1 failed: {e}")
        sys.exit(1)