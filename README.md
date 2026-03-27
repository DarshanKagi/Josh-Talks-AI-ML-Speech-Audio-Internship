# Josh Talks AI/ML Speech & Audio Internship - Complete Solution

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app/)
[![Transformers](https://img.shields.io/badge/Transformers-4.35+-green.svg)](https://huggingface.co/transformers/)

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Task Descriptions](#task-descriptions)
- [Usage Instructions](#usage-instructions)
- [Gradio Demos](#gradio-demos)
- [Output Files](#output-files)
- [Dataset Information](#dataset-information)
- [Results & Metrics](#results--metrics)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## 🎯 Overview

This repository contains the complete solution for the **Josh Talks AI/ML Speech & Audio Internship** technical assignment. The project demonstrates expertise in:

- **Automatic Speech Recognition (ASR)** using Whisper models
- **Hindi Language Processing** with Devanagari script
- **Post-processing Pipelines** for ASR cleanup
- **Spelling Validation** for large-scale datasets
- **Lattice-Based Evaluation** for fair ASR metrics
- **Interactive Web Demos** using Gradio

### Key Achievements

| Task | Focus Area | Key Deliverable |
| :--- | :--- | :--- |
| **Task 1** | Whisper Fine-Tuning | Hindi ASR model with improved WER |
| **Task 2** | ASR Cleanup Pipeline | Number normalization + English word tagging |
| **Task 3** | Spelling Validation | 177K words classified with confidence scores |
| **Task 4** | Lattice Evaluation | Fair WER computation with consensus mechanism |

---

## 📁 Project Structure

```
josh_talks_internship/
│
├── requirements.txt              # All Python dependencies
├── config.py                     # Shared configuration & utilities
├── README.md                     # This documentation file
│
├── task1_finetune.py             # Task 1: Whisper fine-tuning
├── task2_cleanup.py              # Task 2: ASR cleanup pipeline
├── task3_spelling.py             # Task 3: Spelling validation
├── task4_lattice.py              # Task 4: Lattice evaluation
│
├── gradio_task1_demo.py          # Task 1: ASR transcription demo
├── gradio_task2_3_demo.py        # Task 2 & 3: Cleanup + spelling demo
├── gradio_task4_demo.py          # Task 4: Lattice visualization demo
│
├── data/                         # Input data files (provided)
│   ├── FT Data.xlsx              # Task 1: 104 Hindi audio recordings
│   ├── 825780_transcription.json # Task 1: Sample transcription
│   ├── FT Result.xlsx            # Task 1: WER results template
│   ├── Unique Words Data.xlsx    # Task 3: 177,509 unique words
│   └── Question 4.xlsx           # Task 4: 46 segments with 6 transcripts
│
├── models/                       # Trained model weights (generated)
│   └── whisper-hindi-ft/         # Fine-tuned Whisper model
│
├── outputs/                      # Generated results (created on run)
│   ├── FT_Result.xlsx            # Task 1: WER comparison results
│   ├── task1_training_report.json
│   ├── task2_cleanup_examples.csv
│   ├── task3_spelling_classification.csv
│   ├── task4_lattice_evaluation_results.csv
│   └── *.log                     # Execution logs
│
└── verify_installation.py        # Verify all dependencies installed
```

---

## ⚙️ Installation

### Prerequisites

- **Python:** 3.9 - 3.11
- **RAM:** 16 GB minimum (32 GB recommended)
- **GPU:** NVIDIA GPU with 8GB+ VRAM (recommended for Task 1)
- **Storage:** 50 GB free space

### Step-by-Step Setup

```bash
# 1. Clone or download the repository
cd josh_talks_internship

# 2. Create virtual environment (recommended)
python -m venv josh_env

# 3. Activate environment
# Windows:
josh_env\Scripts\activate
# Linux/Mac:
source josh_env/bin/activate

# 4. Install all dependencies
pip install -r requirements.txt

# 5. Verify installation
python verify_installation.py

# 6. (Optional) GPU-specific PyTorch installation
# For CUDA 11.8:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
# For CUDA 12.1:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Verify Installation

```bash
python verify_installation.py
```

Expected output:
```
✅ PyTorch: 2.0.0
✅ Transformers: 4.35.0
✅ Gradio: 4.0.0
✅ jiwer: 3.0.0
✅ Pandas: 2.0.0
✅ Librosa: 0.10.0
✅ CUDA Available: True
✅ All packages installed successfully!
```

---

## 📝 Task Descriptions

### Task 1: Whisper-Small Fine-Tuning on Hindi ASR

**Objective:** Fine-tune OpenAI's Whisper-small model on ~10 hours of Hindi speech data.

**Key Features:**
- Dynamic GCS URL correction (joshtalks-data-collection → upload_goai)
- Multi-segment transcription concatenation from JSON files
- FLEURS Hindi test set evaluation
- WER comparison (baseline vs fine-tuned)

**Files:**
- `task1_finetune.py` - Main training script
- `gradio_task1_demo.py` - Interactive transcription demo

**Expected Output:**
- Fine-tuned model weights in `models/whisper-hindi-ft/`
- WER results in `outputs/FT_Result.xlsx`
- Baseline WER: ~0.83 (83%)
- Target Fine-tuned WER: < 0.50 (50%)

---

### Task 2: ASR Cleanup Pipeline

**Objective:** Build a post-processing pipeline to clean raw ASR output.

**Components:**
1. **Number Normalization:** Hindi words → digits (दो → 2, तीन सौ चौवन → 354)
2. **English Word Detection:** Tag loanwords ([EN]प्रोजेक्ट[/EN])
3. **Idiom Preservation:** Skip idiomatic expressions (दो-चार बातें → unchanged)

**Key Features:**
- Precision-focused (avoid incorrect conversions in idioms)
- Hybrid approach (rule-based + contextual checks)
- WER evaluation before/after cleanup

**Files:**
- `task2_cleanup.py` - Main cleanup pipeline
- `gradio_task2_3_demo.py` - Interactive demo (Task 2 & 3)

**Expected Output:**
- `outputs/task2_cleanup_examples.csv` - Before/after examples
- `outputs/task2_pipeline_statistics.json` - Processing statistics

---

### Task 3: Spelling Validation Pipeline

**Objective:** Classify ~177,000 unique Hindi words as correct/incorrect spelling.

**Multi-Layer Validation:**
1. Dictionary lookup (standard Hindi words)
2. English loanword detection (Devanagari transliterations)
3. Proper noun check (names, places, organizations)
4. Phonetic validation (Devanagari structure)
5. Frequency analysis (word occurrence in corpus)

**Confidence Scoring:**
- **High:** Dictionary match OR invalid structure
- **Medium:** Frequency-based OR single weak signal
- **Low:** Conflicting signals (needs manual review)

**Files:**
- `task3_spelling.py` - Main validation pipeline
- `gradio_task2_3_demo.py` - Interactive demo (Task 2 & 3)

**Expected Output:**
- `outputs/task3_spelling_classification.csv` - All words classified (Google Sheet format)
- `outputs/task3_summary_report.json` - Statistics & unreliable categories
- `outputs/task3_low_confidence_sample.csv` - 50 words for manual review

**Expected Distribution:**
| Category | Count | Percentage |
| :--- | :--- | :--- |
| Correct Spelling | ~145,000-155,000 | ~82-87% |
| Incorrect Spelling | ~15,000-25,000 | ~8-14% |
| Uncertain (Review) | ~5,000-10,000 | ~3-6% |

---

### Task 4: Lattice-Based ASR Evaluation

**Objective:** Build fair evaluation framework using lattice-based WER computation.

**Key Concepts:**
- **Lattice Construction:** Word-level bins with all valid alternatives
- **Consensus Mechanism:** ≥4/5 models can override human reference
- **Best-Path Matching:** Compare against closest valid path (not rigid string)
- **Fair WER:** Reduce penalties for valid alternative transcriptions

**Input:** 46 segments with Human + 5 ASR model outputs (Model H, i, k, l, m, n)

**Files:**
- `task4_lattice.py` - Main lattice evaluation pipeline
- `gradio_task4_demo.py` - Interactive lattice visualization

**Expected Output:**
- `outputs/task4_lattice_evaluation_results.csv` - WER comparison (standard vs lattice)
- `outputs/task4_lattices.json` - Complete lattice structures
- `outputs/task4_summary_report.json` - Aggregate statistics

**Expected Improvement:**
- Average WER Reduction: 15-25%
- Unfairly Penalized Cases Identified: 40-60%

---

## 🚀 Usage Instructions

### Quick Start (Run All Tasks)

```bash
# Task 1: Fine-tune Whisper model (requires GPU, ~2-4 hours)
python task1_finetune.py

# Task 2: Run ASR cleanup pipeline (~5-10 minutes)
python task2_cleanup.py

# Task 3: Run spelling validation (~10-20 minutes)
python task3_spelling.py

# Task 4: Run lattice evaluation (~5-10 minutes)
python task4_lattice.py
```

### Individual Task Commands

#### Task 1: Fine-Tuning

```bash
# Standard run
python task1_finetune.py

# With custom parameters
python task1_finetune.py --epochs 5 --batch_size 4 --learning_rate 1e-5

# Check outputs
ls outputs/
# - FT_Result.xlsx
# - task1_training_report.json
# - task1_training.log
```

#### Task 2: Cleanup Pipeline

```bash
# Standard run
python task2_cleanup.py

# Process specific number of samples
python task2_cleanup.py --samples 20

# Launch Gradio demo only
python task2_cleanup.py --demo
```

#### Task 3: Spelling Validation

```bash
# Standard run
python task3_spelling.py

# Launch manual review interface
python task3_spelling.py --review

# Custom sample size for review
python task3_spelling.py --samples 100
```

#### Task 4: Lattice Evaluation

```bash
# Standard run
python task4_lattice.py

# Custom consensus threshold
python task4_lattice.py --threshold 3

# Launch Gradio demo only
python task4_lattice.py --demo
```

---

## 🎨 Gradio Demos

### Launch All Demos

```bash
# Task 1: ASR Transcription Demo
python gradio_task1_demo.py

# Task 2 & 3: Cleanup + Spelling Demo
python gradio_task2_3_demo.py

# Task 4: Lattice Evaluation Demo
python gradio_task4_demo.py
```

### Demo Features

| Demo | URL | Features |
| :--- | :--- | :--- |
| **Task 1** | http://localhost:7860 | Audio upload, pretrained vs fine-tuned comparison, WER display |
| **Task 2 & 3** | http://localhost:7860 | Text cleanup, number normalization, English tagging, spelling validation |
| **Task 4** | http://localhost:7860 | Lattice visualization, WER comparison, aggregate statistics |

### Public Sharing

```bash
# Create public shareable link (expires in 72 hours)
python gradio_task1_demo.py --share

# Custom port
python gradio_task2_3_demo.py --port 8080
```

---

## 📊 Output Files

### Task 1 Outputs

| File | Description | Format |
| :--- | :--- | :--- |
| `outputs/FT_Result.xlsx` | WER comparison (baseline vs fine-tuned) | Excel |
| `outputs/task1_training_report.json` | Training metrics & configuration | JSON |
| `outputs/task1_training.log` | Full training logs | Log |
| `outputs/task1_failed_loads.json` | Failed audio/transcription loads | JSON |
| `models/whisper-hindi-ft/` | Fine-tuned model weights | Directory |

### Task 2 Outputs

| File | Description | Format |
| :--- | :--- | :--- |
| `outputs/task2_cleanup_comparison.csv` | Before/after cleanup with WER | CSV |
| `outputs/task2_pipeline_statistics.json` | Processing statistics | JSON |
| `outputs/task2_cleanup_examples.csv` | Example conversions | CSV |
| `task2_cleanup.log` | Execution logs | Log |

### Task 3 Outputs

| File | Description | Format |
| :--- | :--- | :--- |
| `outputs/task3_spelling_classification.csv` | All words with classification | CSV |
| `outputs/task3_summary_report.json` | Statistics & unreliable categories | JSON |
| `outputs/task3_low_confidence_sample.csv` | 50 words for manual review | CSV |
| `outputs/task3_manual_review_results.csv` | Human annotation results | CSV |
| `task3_spelling.log` | Execution logs | Log |

### Task 4 Outputs

| File | Description | Format |
| :--- | :--- | :--- |
| `outputs/task4_lattice_evaluation_results.csv` | WER comparison (standard vs lattice) | CSV |
| `outputs/task4_lattices.json` | Complete lattice structures | JSON |
| `outputs/task4_summary_report.json` | Aggregate statistics | JSON |
| `task4_lattice.log` | Execution logs | Log |

---

## 📚 Dataset Information

### Task 1: FT Data (104 Recordings)

| Field | Description | Example |
| :--- | :--- | :--- |
| `user_id` | Anonymized speaker identifier | 245746 |
| `recording_id` | Unique recording identifier | 825780 |
| `language` | Language code | hi (Hindi) |
| `duration` | Audio duration in seconds | 443 |
| `rec_url_gcp` | Audio file URL (GCS) | storage.googleapis.com/... |
| `transcription_url_gcp` | Transcription JSON URL | storage.googleapis.com/... |
| `metadata_url_gcp` | Metadata JSON URL | storage.googleapis.com/... |

**URL Correction Required:**
```
Original: storage.googleapis.com/joshtalks-data-collection/hq_data/hi/{id1}/{id2}_file.ext
Corrected: storage.googleapis.com/upload_goai/{id1}/{id2}_file.ext
```

### Task 3: Unique Words (177,509 Entries)

| Field | Description |
| :--- | :--- |
| `word` | Unique Hindi word in Devanagari script |

**Sample Words:**
- Common: है, तो, और, में, का, की, के
- Loanwords: प्रोजेक्ट, एरिया, टेंट, लाइट
- Proper Nouns: राजस्थान, कोटा, प्रयागराज
- Suspicious: खेतीबाडऱी, मोनता, invalid१२३

### Task 4: Question 4 (46 Segments)

| Field | Description |
| :--- | :--- |
| `segment_url_link` | Audio segment URL |
| `Human` | Human reference transcription |
| `Model H, i, k, l, m, n` | 5 ASR model outputs |

---

## 📈 Results & Metrics

### Task 1: Expected WER Improvement

| Model | WER (Ratio) | WER (%) |
| :--- | :--- | :--- |
| Whisper Small (Pretrained) | 0.83 | 83.0% |
| FT Whisper Small (Ours) | ~0.40-0.60 | 40-60% |
| **Improvement** | **~0.23-0.43** | **~27-52%** |

### Task 2: Cleanup Effectiveness

| Metric | Expected Value |
| :--- | :--- |
| Number Conversions | 5-15 per transcript |
| English Words Detected | 3-10 per transcript |
| Idioms Skipped | 1-3 per transcript |
| WER Improvement | 5-15% |

### Task 3: Classification Distribution

| Category | Count | Percentage |
| :--- | :--- | :--- |
| Correct Spelling | ~145,000-155,000 | ~82-87% |
| Incorrect Spelling | ~15,000-25,000 | ~8-14% |
| Uncertain (Review) | ~5,000-10,000 | ~3-6% |

### Task 4: Lattice WER Improvement

| Metric | Expected Value |
| :--- | :--- |
| Average Standard WER | 0.12-0.18 |
| Average Lattice WER | 0.09-0.14 |
| WER Reduction | 15-25% |
| Unfairly Penalized Cases | 40-60% |
| Reference Override Cases | 15-30 |

---

## 🔧 Troubleshooting

### Common Issues

#### 1. GPU Out of Memory

```bash
# Reduce batch size
python task1_finetune.py --batch_size 2

# Use gradient accumulation
# Edit task1_finetune.py: gradient_accumulation_steps=4
```

#### 2. GCS URL Access Denied

```python
# URLs are automatically corrected in config.py
# Verify correction:
from config import correct_gcs_url
url = correct_gcs_url("https://storage.googleapis.com/joshtalks-data-collection/...")
print(url)
```

#### 3. Gradio Port Already in Use

```bash
# Use different port
python gradio_task1_demo.py --port 8080
```

#### 4. Missing Dependencies

```bash
# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall

# Verify installation
python verify_installation.py
```

#### 5. Hindi Text Display Issues

```bash
# Ensure UTF-8 encoding
export PYTHONIOENCODING=utf-8  # Linux/Mac
set PYTHONIOENCODING=utf-8     # Windows
```

### Getting Help

1. Check log files in `outputs/*.log`
2. Review error messages in console output
3. Verify all dependencies are installed correctly
4. Ensure input data files are in `data/` directory

---

## 📄 License

This project is created for the **Josh Talks AI/ML Speech & Audio Internship** technical assignment.

- **Usage:** Educational and evaluation purposes only
- **Data:** Provided by Josh Talks (confidential)
- **Models:** Whisper-small (OpenAI Apache 2.0 License)

---

## 👥 Contact & Support

### Project Information

- **Internship:** Josh Talks AI/ML Speech & Audio
- **Role:** AI/ML Engineer Intern
- **Tasks:** 4 (ASR Fine-tuning, Cleanup, Spelling, Lattice Evaluation)

### File Checklist for Submission

```
✅ requirements.txt
✅ config.py
✅ task1_finetune.py
✅ task2_cleanup.py
✅ task3_spelling.py
✅ task4_lattice.py
✅ gradio_task1_demo.py
✅ gradio_task2_3_demo.py
✅ gradio_task4_demo.py
✅ README.md
✅ outputs/FT_Result.xlsx (filled)
✅ outputs/task3_spelling_classification.csv
✅ All Gradio demos working
```

---

## 🎉 Quick Start Summary

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all tasks
python task1_finetune.py
python task2_cleanup.py
python task3_spelling.py
python task4_lattice.py

# 3. Launch demos
python gradio_task1_demo.py
python gradio_task2_3_demo.py
python gradio_task4_demo.py

# 4. Check outputs
ls outputs/
```

---

**Built with ❤️ for Josh Talks AI/ML Speech & Audio Internship**

*Last Updated: 2024*
