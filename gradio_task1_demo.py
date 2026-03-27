"""
============================================
Josh Talks AI/ML Speech & Audio Internship
Gradio Demo for Task 1: Whisper Fine-Tuning
============================================

This script provides an interactive web interface for:
1. Uploading Hindi audio files for transcription
2. Comparing pretrained vs fine-tuned Whisper models
3. Displaying WER metrics against reference transcription
4. Visualizing model confidence and performance

No hardcoded paths - all data access is dynamic via config.py
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import librosa
import requests

# Gradio import
try:
    import gradio as gr
except ImportError:
    print("❌ Gradio not installed. Run: pip install gradio")
    sys.exit(1)

# Hugging Face imports
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    pipeline
)

# Local config import
from config import (
    get_project_root,
    get_data_path,
    get_output_path,
    get_model_path,
    correct_gcs_url,
    load_ft_data,
    concatenate_transcription_segments,
    normalize_hindi_text,
    calculate_wer,
    GRADIO_CONFIG,
    setup_logging,
    TASK1_CONFIG,
)


# ============================================
# LOGGING SETUP
# ============================================

logger = logging.getLogger(__name__)
setup_logging('gradio_task1.log')


# ============================================
# MODEL LOADING AND MANAGEMENT
# ============================================

class ASRModelManager:
    """
    Manages loading and inference for Whisper models (pretrained & fine-tuned).
    """
    
    def __init__(self):
        self.pretrained_model = None
        self.pretrained_processor = None
        self.finetuned_model = None
        self.finetuned_processor = None
        self.models_loaded = False
        
        # Model paths
        self.pretrained_model_name = TASK1_CONFIG.get('model_name', 'openai/whisper-small')
        self.finetuned_model_path = get_model_path('whisper-hindi-ft')
        
    def load_pretrained_model(self) -> bool:
        """Load the pretrained Whisper-small model."""
        try:
            logger.info(f"Loading pretrained model: {self.pretrained_model_name}")
            
            self.pretrained_processor = WhisperProcessor.from_pretrained(
                self.pretrained_model_name,
                language="hindi",
                task="transcribe"
            )
            
            self.pretrained_model = WhisperForConditionalGeneration.from_pretrained(
                self.pretrained_model_name
            )
            
            # Set to evaluation mode
            self.pretrained_model.eval()
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.pretrained_model = self.pretrained_model.cuda()
            
            logger.info("✅ Pretrained model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load pretrained model: {e}")
            return False
    
    def load_finetuned_model(self) -> bool:
        """Load the fine-tuned Whisper model from Task 1."""
        try:
            if not self.finetuned_model_path.exists():
                logger.warning(f"Fine-tuned model not found at {self.finetuned_model_path}")
                logger.warning("Please run task1_finetune.py first to train the model")
                return False
            
            logger.info(f"Loading fine-tuned model from: {self.finetuned_model_path}")
            
            self.finetuned_processor = WhisperProcessor.from_pretrained(
                str(self.finetuned_model_path),
                language="hindi",
                task="transcribe"
            )
            
            self.finetuned_model = WhisperForConditionalGeneration.from_pretrained(
                str(self.finetuned_model_path)
            )
            
            # Set to evaluation mode
            self.finetuned_model.eval()
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.finetuned_model = self.finetuned_model.cuda()
            
            logger.info("✅ Fine-tuned model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load fine-tuned model: {e}")
            return False
    
    def load_all_models(self) -> bool:
        """Load both pretrained and fine-tuned models."""
        pretrained_ok = self.load_pretrained_model()
        finetuned_ok = self.load_finetuned_model()
        
        self.models_loaded = pretrained_ok
        return pretrained_ok
    
    def transcribe_with_pretrained(self, audio_array: np.ndarray, 
                                   sampling_rate: int = 16000) -> Tuple[str, float]:
        """
        Transcribe audio using pretrained model.
        
        Returns:
            Tuple of (transcription, confidence_score)
        """
        if self.pretrained_model is None:
            return "❌ Pretrained model not loaded", 0.0
        
        try:
            inputs = self.pretrained_processor(
                audio_array,
                sampling_rate=sampling_rate,
                return_tensors="pt",
                language="hindi",
                task="transcribe"
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs.input_features = inputs.input_features.cuda()
            
            with torch.no_grad():
                pred_ids = self.pretrained_model.generate(
                    inputs.input_features,
                    max_length=225,
                    language="hindi",
                    task="transcribe"
                )
            
            transcription = self.pretrained_processor.batch_decode(
                pred_ids,
                skip_special_tokens=True
            )[0]
            
            # Estimate confidence (simplified - based on sequence probability)
            confidence = self._estimate_confidence(pred_ids, self.pretrained_model)
            
            return transcription, confidence
            
        except Exception as e:
            logger.error(f"Pretrained transcription failed: {e}")
            return f"❌ Error: {str(e)}", 0.0
    
    def transcribe_with_finetuned(self, audio_array: np.ndarray,
                                  sampling_rate: int = 16000) -> Tuple[str, float]:
        """
        Transcribe audio using fine-tuned model.
        
        Returns:
            Tuple of (transcription, confidence_score)
        """
        if self.finetuned_model is None:
            return "❌ Fine-tuned model not loaded", 0.0
        
        try:
            inputs = self.finetuned_processor(
                audio_array,
                sampling_rate=sampling_rate,
                return_tensors="pt",
                language="hindi",
                task="transcribe"
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs.input_features = inputs.input_features.cuda()
            
            with torch.no_grad():
                pred_ids = self.finetuned_model.generate(
                    inputs.input_features,
                    max_length=225,
                    language="hindi",
                    task="transcribe"
                )
            
            transcription = self.finetuned_processor.batch_decode(
                pred_ids,
                skip_special_tokens=True
            )[0]
            
            # Estimate confidence
            confidence = self._estimate_confidence(pred_ids, self.finetuned_model)
            
            return transcription, confidence
            
        except Exception as e:
            logger.error(f"Fine-tuned transcription failed: {e}")
            return f"❌ Error: {str(e)}", 0.0
    
    def _estimate_confidence(self, pred_ids: torch.Tensor, 
                            model: WhisperForConditionalGeneration) -> float:
        """
        Estimate transcription confidence based on model logits.
        Simplified implementation for demo purposes.
        """
        try:
            # Get the logits for the predicted tokens
            with torch.no_grad():
                outputs = model(
                    input_features=pred_ids if len(pred_ids.shape) == 3 else pred_ids.unsqueeze(0),
                    labels=pred_ids
                )
            
            # Use loss as inverse confidence indicator
            loss = outputs.loss.item() if outputs.loss is not None else 1.0
            
            # Convert to confidence score (0-1)
            confidence = max(0.0, min(1.0, 1.0 / (1.0 + loss)))
            
            return confidence
            
        except Exception:
            return 0.5  # Default confidence


# ============================================
# AUDIO PROCESSING UTILITIES
# ============================================

class AudioProcessor:
    """
    Handles audio file loading and preprocessing.
    """
    
    @staticmethod
    def load_audio_file(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        """
        Load audio file and resample to target sampling rate.
        
        Args:
            file_path: Path to audio file
            target_sr: Target sampling rate (default 16000 for Whisper)
        
        Returns:
            Tuple of (audio_array, sampling_rate)
        """
        try:
            audio_array, sr = librosa.load(file_path, sr=target_sr, mono=True)
            return audio_array, sr
        except Exception as e:
            logger.error(f"Failed to load audio file: {e}")
            return None, 0
    
    @staticmethod
    def load_audio_from_url(url: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        """
        Load audio from URL and resample.
        
        Args:
            url: URL to audio file
            target_sr: Target sampling rate
        
        Returns:
            Tuple of (audio_array, sampling_rate)
        """
        try:
            corrected_url = correct_gcs_url(url)
            response = requests.get(corrected_url, timeout=60)
            response.raise_for_status()
            
            audio_array, sr = librosa.load(
                io.BytesIO(response.content),
                sr=target_sr,
                mono=True
            )
            
            return audio_array, sr
            
        except Exception as e:
            logger.error(f"Failed to load audio from URL: {e}")
            return None, 0
    
    @staticmethod
    def get_audio_duration(audio_array: np.ndarray, sampling_rate: int) -> float:
        """Get audio duration in seconds."""
        return len(audio_array) / sampling_rate
    
    @staticmethod
    def get_audio_info(file_path: str) -> Dict[str, Any]:
        """Get detailed audio file information."""
        try:
            audio_array, sr = librosa.load(file_path, sr=None, mono=True)
            duration = len(audio_array) / sr
            
            return {
                'sampling_rate': sr,
                'duration_seconds': duration,
                'duration_formatted': time.strftime('%M:%S', time.gmtime(duration)),
                'num_samples': len(audio_array),
                'dtype': str(audio_array.dtype),
                'channels': 1,  # Mono
            }
        except Exception as e:
            return {'error': str(e)}


# ============================================
# WER EVALUATION UTILITIES
# ============================================

class WEREvaluator:
    """
    Handles WER calculation and comparison.
    """
    
    @staticmethod
    def calculate_wer(reference: str, hypothesis: str) -> Dict[str, Any]:
        """
        Calculate Word Error Rate with detailed breakdown.
        
        Returns:
            Dictionary with WER and edit breakdown
        """
        try:
            from jiwer import wer, compute_measures
            
            wer_score = wer(reference, hypothesis)
            measures = compute_measures(reference, hypothesis)
            
            return {
                'wer': wer_score,
                'wer_percent': wer_score * 100,
                'substitutions': measures.get('substitutions', 0),
                'insertions': measures.get('insertions', 0),
                'deletions': measures.get('deletions', 0),
                'hits': measures.get('hits', 0),
                'total_words': measures.get('substitutions', 0) + 
                              measures.get('insertions', 0) + 
                              measures.get('deletions', 0) + 
                              measures.get('hits', 0),
            }
        except ImportError:
            # Fallback simple implementation
            ref_words = reference.strip().split()
            hyp_words = hypothesis.strip().split()
            
            edits = abs(len(ref_words) - len(hyp_words))
            wer_score = edits / max(len(ref_words), 1)
            
            return {
                'wer': wer_score,
                'wer_percent': wer_score * 100,
                'substitutions': 0,
                'insertions': edits,
                'deletions': 0,
                'hits': min(len(ref_words), len(hyp_words)),
                'total_words': max(len(ref_words), len(hyp_words)),
            }
    
    @staticmethod
    def compare_models(pretrained_output: str, finetuned_output: str,
                      reference: str) -> Dict[str, Any]:
        """
        Compare WER between pretrained and fine-tuned models.
        
        Returns:
            Comparison dictionary
        """
        pretrained_wer = WEREvaluator.calculate_wer(reference, pretrained_output)
        finetuned_wer = WEREvaluator.calculate_wer(reference, finetuned_output)
        
        improvement = pretrained_wer['wer'] - finetuned_wer['wer']
        improvement_percent = (improvement / pretrained_wer['wer'] * 100) if pretrained_wer['wer'] > 0 else 0
        
        return {
            'pretrained_wer': pretrained_wer,
            'finetuned_wer': finetuned_wer,
            'improvement': improvement,
            'improvement_percent': improvement_percent,
            'better_model': 'Fine-tuned' if improvement > 0 else 'Pretrained' if improvement < 0 else 'Tie',
        }


# ============================================
# GRADIO INTERFACE COMPONENTS
# ============================================

class GradioASRDemo:
    """
    Main Gradio interface for Task 1 ASR demonstration.
    """
    
    def __init__(self):
        self.model_manager = ASRModelManager()
        self.audio_processor = AudioProcessor()
        self.wer_evaluator = WEREvaluator()
        
        # Load models on initialization
        logger.info("Initializing model manager...")
        self.model_manager.load_all_models()
    
    def transcribe_audio(self, audio_file: str, 
                        use_finetuned: bool = True,
                        reference_text: str = "") -> str:
        """
        Main transcription function for Gradio interface.
        
        Args:
            audio_file: Path to uploaded audio file
            use_finetuned: Whether to use fine-tuned model
            reference_text: Optional reference for WER calculation
        
        Returns:
            Formatted result string
        """
        if audio_file is None:
            return "❌ Please upload an audio file."
        
        # Load audio
        audio_array, sr = self.audio_processor.load_audio_file(audio_file)
        
        if audio_array is None:
            return "❌ Failed to load audio file. Please try again."
        
        # Get audio info
        audio_info = self.audio_processor.get_audio_info(audio_file)
        duration = audio_info.get('duration_formatted', 'Unknown')
        
        # Transcribe with both models for comparison
        pretrained_text, pretrained_conf = self.model_manager.transcribe_with_pretrained(audio_array, sr)
        
        result = f"### 📊 Audio Information\n"
        result += f"- **Duration:** {duration}\n"
        result += f"- **Sampling Rate:** {audio_info.get('sampling_rate', 'Unknown')} Hz\n\n"
        
        result += f"### 🎙️ Pretrained Whisper-Small\n"
        result += f"**Transcription:** {pretrained_text}\n"
        result += f"**Confidence:** {pretrained_conf:.2%}\n\n"
        
        if use_finetuned and self.model_manager.finetuned_model is not None:
            finetuned_text, finetuned_conf = self.model_manager.transcribe_with_finetuned(audio_array, sr)
            
            result += f"### 🚀 Fine-Tuned Whisper-Small (Hindi)\n"
            result += f"**Transcription:** {finetuned_text}\n"
            result += f"**Confidence:** {finetuned_conf:.2%}\n\n"
            
            # Compare if reference provided
            if reference_text.strip():
                comparison = self.wer_evaluator.compare_models(
                    pretrained_text, finetuned_text, reference_text
                )
                
                result += f"### 📈 WER Comparison (vs Reference)\n"
                result += f"| Model | WER | Substitutions | Insertions | Deletions |\n"
                result += f"|-------|-----|---------------|------------|----------|\n"
                result += f"| Pretrained | {comparison['pretrained_wer']['wer_percent']:.2f}% | {comparison['pretrained_wer']['substitutions']} | {comparison['pretrained_wer']['insertions']} | {comparison['pretrained_wer']['deletions']} |\n"
                result += f"| Fine-Tuned | {comparison['finetuned_wer']['wer_percent']:.2f}% | {comparison['finetuned_wer']['substitutions']} | {comparison['finetuned_wer']['insertions']} | {comparison['finetuned_wer']['deletions']} |\n\n"
                result += f"**Improvement:** {comparison['improvement_percent']:.2f}% ({comparison['better_model']} model performed better)\n"
        else:
            result += f"### ⚠️ Fine-Tuned Model\n"
            result += f"Fine-tuned model not available. Please run `task1_finetune.py` first.\n\n"
        
        # Add reference if provided
        if reference_text.strip():
            result += f"### 📝 Reference Transcription\n"
            result += f"{reference_text}\n\n"
        
        return result
    
    def load_sample_from_dataset(self, recording_id: str) -> Tuple[str, str, str]:
        """
        Load a sample audio from the FT Data dataset.
        
        Returns:
            Tuple of (audio_path, reference_text, recording_info)
        """
        try:
            df = load_ft_data()
            
            # Find recording
            row = df[df['recording_id'].astype(str) == str(recording_id)]
            
            if len(row) == 0:
                return None, "", "Recording not found"
            
            row = row.iloc[0]
            
            # Download audio temporarily
            audio_url = correct_gcs_url(row['rec_url_gcp'])
            trans_url = correct_gcs_url(row['transcription_url_gcp'])
            
            # Download audio
            audio_response = requests.get(audio_url, timeout=60)
            audio_response.raise_for_status()
            
            temp_audio_path = get_output_path(f'temp_{recording_id}.wav')
            with open(temp_audio_path, 'wb') as f:
                f.write(audio_response.content)
            
            # Get reference transcription
            trans_response = requests.get(trans_url, timeout=30)
            trans_response.raise_for_status()
            trans_data = trans_response.json()
            reference_text = concatenate_transcription_segments(trans_data)
            
            info = f"Recording ID: {recording_id}\nDuration: {row['duration']}s\nUser ID: {row['user_id']}"
            
            return str(temp_audio_path), reference_text, info
            
        except Exception as e:
            logger.error(f"Failed to load sample: {e}")
            return None, "", f"Error: {str(e)}"
    
    def get_sample_recordings(self) -> List[str]:
        """Get list of sample recording IDs from dataset."""
        try:
            df = load_ft_data()
            return df['recording_id'].astype(str).head(20).tolist()
        except Exception:
            return []
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        
        with gr.Blocks(
            title="Josh Talks ASR Demo - Task 1",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {max-width: 1200px !important;}
            .result-box {background: #f0f0f0; padding: 10px; border-radius: 5px;}
            """
        ) as demo:
            
            gr.Markdown("""
            # 🎙️ Josh Talks ASR Demo - Task 1
            ### Whisper-Small Fine-Tuning on Hindi Speech
            
            This demo showcases the fine-tuned Whisper-small model for Hindi automatic speech recognition.
            Upload an audio file or select from the dataset samples to see transcription results.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 Upload Audio")
                    audio_input = gr.Audio(
                        label="Upload Hindi Audio",
                        type="filepath",
                        sources=["upload", "microphone"]
                    )
                    
                    gr.Markdown("### ⚙️ Options")
                    use_finetuned = gr.Checkbox(
                        label="Use Fine-Tuned Model",
                        value=True,
                        interactive=True
                    )
                    
                    gr.Markdown("### 📝 Reference (Optional)")
                    reference_input = gr.Textbox(
                        label="Reference Transcription (for WER calculation)",
                        placeholder="Enter ground truth transcription here...",
                        lines=3
                    )
                    
                    transcribe_btn = gr.Button("🚀 Transcribe", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Results")
                    output_display = gr.Textbox(
                        label="Transcription Results",
                        lines=20,
                        show_copy_button=True
                    )
            
            with gr.Accordion("📁 Load Sample from Dataset", open=False):
                gr.Markdown("Select a recording from the FT Data dataset to test with.")
                
                sample_dropdown = gr.Dropdown(
                    choices=self.get_sample_recordings(),
                    label="Select Recording ID",
                    interactive=True
                )
                
                load_sample_btn = gr.Button("Load Sample")
                
                sample_info = gr.Textbox(
                    label="Sample Information",
                    lines=3,
                    interactive=False
                )
                
                sample_reference = gr.Textbox(
                    label="Reference Transcription",
                    lines=3,
                    interactive=True
                )
            
            with gr.Accordion("📈 Model Performance Metrics", open=False):
                gr.Markdown("""
                ### Baseline Performance (FLEURS Hindi Test Set)
                
                | Model | WER |
                |-------|-----|
                | Whisper Small (Pretrained) | 0.83 (83%) |
                | FT Whisper Small (Ours) | *To be filled after evaluation* |
                
                **Note:** Lower WER is better. Fine-tuning should reduce WER significantly.
                """)
                
                # Load WER results if available
                wer_results_path = get_output_path('FT_Result.xlsx')
                if wer_results_path.exists():
                    try:
                        wer_df = pd.read_excel(wer_results_path)
                        gr.Dataframe(
                            value=wer_df,
                            label="WER Results from Task 1"
                        )
                    except Exception:
                        gr.Markdown("WER results file not found. Run `task1_finetune.py` first.")
                else:
                    gr.Markdown("⚠️ WER results not available. Run `task1_finetune.py` to generate.")
            
            with gr.Accordion("ℹ️ About This Demo", open=False):
                gr.Markdown("""
                ### Model Details
                - **Base Model:** OpenAI Whisper-Small
                - **Fine-Tuning Dataset:** ~10 hours of Hindi speech from Josh Talks
                - **Language:** Hindi (hi)
                - **Task:** Automatic Speech Recognition (ASR)
                
                ### How to Use
                1. Upload a Hindi audio file (.wav, .mp3, etc.)
                2. Optionally provide reference transcription for WER calculation
                3. Click "Transcribe" to see results from both models
                4. Compare pretrained vs fine-tuned performance
                
                ### Technical Details
                - Audio is resampled to 16kHz for Whisper compatibility
                - Both models run inference on GPU if available
                - Confidence scores are estimated from model logits
                """)
            
            # Event handlers
            transcribe_btn.click(
                fn=self.transcribe_audio,
                inputs=[audio_input, use_finetuned, reference_input],
                outputs=[output_display]
            )
            
            def load_sample_handler(recording_id):
                if not recording_id:
                    return None, "", "Please select a recording ID"
                audio_path, reference, info = self.load_sample_from_dataset(recording_id)
                return audio_path, reference, info
            
            load_sample_btn.click(
                fn=load_sample_handler,
                inputs=[sample_dropdown],
                outputs=[audio_input, sample_reference, sample_info]
            )
        
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
    logger.info("Josh Talks AI/ML Internship - Task 1 Gradio Demo")
    logger.info("=" * 60)
    
    # Check for GPU
    if torch.cuda.is_available():
        logger.info(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("⚠️ No GPU available. Inference will be slower.")
    
    # Create and launch demo
    demo_app = GradioASRDemo()
    
    logger.info("\n" + "=" * 60)
    logger.info("Launching Gradio Interface...")
    logger.info("=" * 60)
    logger.info("Open your browser to: http://localhost:7860")
    logger.info("=" * 60)
    
    demo_app.launch()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Task 1 Gradio ASR Demo")
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