# Multilingual-ASR-Evaluation

A reproducible, modular Python evaluation framework and interactive dashboard for benchmarking locally executed, open-source **OpenAI Whisper** models (`tiny` and `base`) across **Hindi**, **Marathi**, and **English** speech recognition.

---

## 1. Project Overview

**Multilingual-ASR-Evaluation** provides an end-to-end pipeline for validating multilingual speech datasets, running local OpenAI Whisper transcription models without external APIs or API keys, normalizing Unicode transcripts safely for Devanagari and Latin scripts, and computing standardized Word Error Rate (WER), Character Error Rate (CER), processing latency, and Real-Time Factor (RTF).

Results are exported to reproducible CSV and JSON artifacts and visualized through both a command-line interface (CLI) and an interactive Streamlit dashboard.

---

## 2. Problem Statement

Evaluating multilingual Automatic Speech Recognition (ASR) models—especially for low- and mid-resource Indic languages such as Hindi and Marathi alongside English—requires:

1. **Controlled, paired benchmarking** on identical audio clips and verified reference transcripts across model sizes.
2. **Script-aware Unicode handling** so Devanagari characters, vowel signs (*matras*), *anusvara*, *chandrabindu*, and *virama* are never corrupted during normalization or metric scoring.
3. **Transparent, local-only execution** without paid API dependencies, silent metric fabrication, or unverified placeholder results.

---

## 3. Features

- **Local OpenAI Whisper Execution**: Runs `tiny` and `base` models locally with automatic weight download and in-memory caching.
- **Multilingual Language Mapping**: Explicit language conditioning (`Hindi -> hi`, `Marathi -> mr`, `English -> en`) instead of relying solely on automatic language detection.
- **Hardware-Aware Precision**: Automatically selects CUDA (FP16) when available or CPU (FP32), with explicit `--device` override support.
- **Strict Manifest Validation**: Validates CSV schema, relative paths from project root, supported audio formats (`.wav`, `.mp3`, `.m4a`, `.flac`), supported languages, and non-empty reference transcripts.
- **Devanagari-Safe Text Normalization**: Configurable NFC Unicode normalization, whitespace normalization, optional punctuation removal (preserving all Devanagari letters and combining marks), and optional Latin lowercasing—explicitly documented in evaluation summaries.
- **Standardized Error & Efficiency Metrics**: Computes WER and CER using `jiwer` (reported as both rates and percentages), audio duration, latency, and Real-Time Factor (RTF).
- **Resilient Batch Pipeline**: Isolates per-file or per-model errors so a single corrupted clip does not abort the entire run.
- **Command-Line Interface & Streamlit Dashboard**: Full CLI (`python -m app.cli`) and interactive web dashboard (`streamlit run streamlit_app/app.py`).
- **Automated Testing & Linting**: Deterministic `pytest` suite (zero model downloads required) and `ruff` static checks.

---

## 4. Technologies Used

- **Python 3.10+**
- **OpenAI Whisper (`openai-whisper`)** & **PyTorch (`torch`)** — local ASR inference
- **`jiwer`** — standardized WER and CER computation
- **`pandas`** — manifest loading, validation, and CSV result export
- **`soundfile`** & **`librosa`** — audio file validation and duration extraction
- **`streamlit`** — interactive web dashboard
- **`plotly`** & **`matplotlib`** — evaluation visualizations
- **`pytest`** — unit and validation testing
- **`ruff`** — Python linting and formatting checks
- **FFmpeg** (system dependency) — required by Whisper/Librosa when decoding compressed audio formats such as MP3 and M4A

---

## 5. Architecture

```
                 +----------------------+
                 |   data/manifest.csv  |
                 |   + data/audio/*     |
                 +----------+-----------+
                            |
                            v
              +-------------+-------------+
              | app/data/manifest_loader  |
              | (Schema, Path & Format    |
              |  Validation)              |
              +-------------+-------------+
                            |
                            v
+-------------------+   +---+-------------------+   +------------------------+
| app/asr/          |   | app/evaluation/       |   | app/evaluation/        |
| model_manager.py  +-->+ evaluator.py          +<--+ normalizer.py          |
| whisper_engine.py |   | (Paired Tiny vs Base) |   | metrics.py (jiwer)     |
+-------------------+   +---+-------------------+   +------------------------+
                            |
             +--------------+--------------+
             |                             |
             v                             v
+------------+-------------+  +------------+-------------+
| results/                 |  | Interfaces               |
| - evaluation_results.csv |  | - app/cli.py             |
| - evaluation_summary.json|  | - streamlit_app/app.py   |
+--------------------------+  +--------------------------+
```

---

## 6. Complete Project Directory Structure

```text
Multilingual-ASR-Evaluation/
│
├── app/
│   ├── __init__.py
│   ├── asr/
│   │   ├── __init__.py
│   │   ├── whisper_engine.py
│   │   └── model_manager.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── normalizer.py
│   │   └── evaluator.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── manifest_loader.py
│   │
│   ├── cli.py
│   └── config.py
│
├── streamlit_app/
│   └── app.py
│
├── data/
│   ├── manifest.csv
│   └── audio/
│       └── .gitkeep
│
├── results/
│   └── .gitkeep
│
├── tests/
│   ├── __init__.py
│   ├── test_metrics.py
│   ├── test_manifest_loader.py
│   └── test_normalizer.py
│
├── scripts/
│   └── run_evaluation.py
│
├── .gitignore
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 7. Installation Instructions

### Prerequisites

- **Python**: Version `3.10` or newer
- **FFmpeg**: Must be installed separately on your system if you plan to evaluate compressed formats (`MP3`, `M4A`) or run OpenAI Whisper decoding:
  - **Ubuntu / Debian**: `sudo apt-get update && sudo apt-get install -y ffmpeg`
  - **macOS (Homebrew)**: `brew install ffmpeg`
  - **Windows**: Install via `winget install Gyan.FFmpeg` or add FFmpeg `bin/` to `PATH`.

---

## 8. Virtual Environment Setup

Create and activate a virtual environment from the project root:

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

---

## 9. Dependency Installation

Upgrade `pip` and install all dependencies from `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*(Optional for CPU-only machines)*: If you do not have an NVIDIA GPU, you may install CPU-only PyTorch first to reduce download size before running `pip install -r requirements.txt`:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
```

---

## 10. Whisper Model Download and Caching Explanation

- This project uses the local `openai-whisper` Python library (not the paid OpenAI cloud API).
- On the first evaluation run for `tiny` or `base`, `whisper.load_model()` automatically downloads the model checkpoint weights into your local cache directory (by default `~/.cache/whisper`, or a custom directory set via the `WHISPER_CACHE_DIR` environment variable).
- Subsequent runs reuse the cached weights on disk, and `ModelManager` caches loaded model objects in memory so models are not reloaded between audio clips.
- No model weights are committed to this repository.

---

## 11. How to Add Custom Audio Files

Follow this workflow to evaluate your own multilingual speech clips:

1. Place your own audio clips (`.wav`, `.mp3`, `.m4a`, or `.flac`) inside `data/audio/`.
2. Add one row per audio file in `data/manifest.csv`.
3. Provide the exact language (`Hindi`, `Marathi`, or `English`).
4. Enter the exact human-verified reference transcript.
5. Run the evaluation command (`python -m app.cli evaluate`).
6. Inspect the generated results in `results/evaluation_results.csv`, `results/evaluation_summary.json`, or the Streamlit dashboard.

> **Important Note on Devanagari Transcripts & Reference Quality:**
> Hindi and Marathi reference transcripts **must use accurate Devanagari spelling** (not Romanized transliteration). Reference transcript quality directly determines the reliability of WER and CER—typos, missing words, or mismatched orthography in the reference will inflate reported error rates.

---

## 12. How to Prepare the Manifest CSV

`data/manifest.csv` must contain UTF-8 encoded CSV data with these exact header columns:

```csv
audio_path,language,reference_text
```

- `audio_path`: Relative path from the project root (e.g., `data/audio/clip_01.wav`).
- `language`: Exactly one of `Hindi`, `Marathi`, or `English`.
- `reference_text`: Non-empty verified transcript in the target script. Quote fields containing commas.

You can validate your manifest without running Whisper models:

```bash
python -m app.cli validate --manifest data/manifest.csv
```

---

## 13. How to Run the CLI

View all available commands and options:

```bash
python -m app.cli --help
python -m app.cli evaluate --help
```

Run evaluation with default settings (`tiny` and `base` on `data/manifest.csv`):

```bash
python -m app.cli evaluate
```

Specify a custom manifest path:

```bash
python -m app.cli evaluate --manifest data/manifest.csv
```

Specify models explicitly:

```bash
python -m app.cli evaluate --models tiny base
```

Select execution device (`cpu`, `cuda`, or `auto`):

```bash
python -m app.cli evaluate --device cpu
```

Filter evaluation to a single language:

```bash
python -m app.cli evaluate --language Marathi
```

Optional normalization flags:

```bash
python -m app.cli evaluate --remove-punctuation --lowercase
python -m app.cli evaluate --no-normalize
```

You can also run the convenience script:

```bash
python scripts/run_evaluation.py --manifest data/manifest.csv --models tiny base --device cpu
```

---

## 14. How to Launch the Streamlit Application

Start the interactive dashboard from the project root:

```bash
streamlit run streamlit_app/app.py
```

The dashboard allows you to upload a manifest CSV, upload individual audio clips with reference transcripts, validate the dataset, configure Whisper Tiny/Base and device settings, run evaluations, inspect Devanagari and English transcripts side-by-side, explore Plotly comparison charts, and export CSV/JSON reports.

---

## 15. How to Run Pytest

Run the full automated test suite (uses temporary files and in-memory fixtures; does not download Whisper models):

```bash
pytest
```

---

## 16. How to Run Ruff

Check code quality, style, and import sorting with Ruff:

```bash
ruff check .
```

---

## 17. Explanation of Word Error Rate (WER)

**Word Error Rate (WER)** measures word-level transcription accuracy based on the Levenshtein edit distance between the reference transcript and the model hypothesis:

$$\text{WER} = \frac{S + D + I}{N}$$

Where:
- $S$ = number of word **substitutions**
- $D$ = number of word **deletions**
- $I$ = number of word **insertions**
- $N$ = total number of words in the **reference** transcript

**Interpretation**: Lower WER indicates better accuracy (`0.0` / `0%` is a perfect match). Note that WER is an **error rate**, not an accuracy percentage—heavy insertions can cause WER to exceed `1.0` (`100%`).

---

## 18. Explanation of Character Error Rate (CER)

**Character Error Rate (CER)** applies the same edit-distance formula at the **character level** rather than the word level:

$$\text{CER} = \frac{S_c + D_c + I_c}{N_c}$$

CER is especially informative for morphologically rich Indic languages such as **Hindi** and **Marathi**, where a single *matra* or suffix difference marks an entire word wrong in WER even when most characters were recognized accurately. Lower CER indicates better character-level fidelity.

---

## 19. Explanation of Real-Time Factor (RTF)

**Real-Time Factor (RTF)** measures inference speed relative to audio duration:

$$\text{RTF} = \frac{\text{Processing Time (seconds)}}{\text{Audio Duration (seconds)}}$$

- $\text{RTF} < 1.0$: Faster than real-time (e.g., $\text{RTF} = 0.25$ means a 10-second clip is transcribed in 2.5 seconds).
- $\text{RTF} = 1.0$: Real-time speed.
- $\text{RTF} > 1.0$: Slower than real-time.

---

## 20. Model Comparison Methodology

1. Both **Whisper Tiny** and **Whisper Base** are evaluated on the **exact same audio clips and reference transcripts**.
2. Each clip is conditioned on its manifest language (`hi`, `mr`, or `en`) so accuracy differences reflect recognition quality rather than language-identification failures.
3. Identical text normalization settings are applied to both models' outputs and documented in `evaluation_summary.json`.
4. Paired deltas (`base - tiny`) for mean WER, mean CER, and processing latency are computed over common successfully evaluated clips.

---

## 21. Output File Descriptions

### `results/evaluation_results.csv`
Per-clip evaluation table containing:
- `audio_filename`, `audio_path`, `language`, `reference_transcript`
- `tiny_transcription`, `base_transcription`
- `tiny_wer`, `base_wer`, `tiny_wer_pct`, `base_wer_pct`
- `tiny_cer`, `base_cer`, `tiny_cer_pct`, `base_cer_pct`
- `tiny_processing_time_sec`, `base_processing_time_sec`, `audio_duration_sec`
- `tiny_rtf`, `base_rtf`
- `status` (`success`, `partial`, or `failed`), `error_message`

### `results/evaluation_summary.json`
Structured run metadata and aggregate statistics including timestamp, device, normalization configuration, overall per-model metrics, language-wise metrics (`Hindi`, `Marathi`, `English`), and paired Tiny vs. Base comparison deltas.

---

## Experimental Results

**Status: Pending actual evaluation.**

No experimental results have been recorded yet.

| Language | Model | WER | CER | Processing Time |
|---|---|---|---|---|
| Hindi | Tiny | Pending | Pending | Pending |
| Hindi | Base | Pending | Pending | Pending |
| Marathi | Tiny | Pending | Pending | Pending |
| Marathi | Base | Pending | Pending | Pending |
| English | Tiny | Pending | Pending | Pending |
| English | Base | Pending | Pending | Pending |

Replace these placeholders only after running the application on actual audio clips with verified reference transcripts.

---

## 22. Limitations

- **CPU Latency**: Running Whisper on CPU (FP32) is slower than GPU inference (FP16).
- **Compact Model Capacity**: `tiny` (~39M parameters) and `base` (~74M parameters) trade multilingual accuracy—especially for accented or noisy Hindi and Marathi speech—for speed and small memory footprint compared to `small`, `medium`, or `large-v3`.
- **Orthographic Sensitivity**: Strict WER/CER penalizes valid spelling variants, punctuation differences (unless `--remove-punctuation` is enabled), and numeral formatting (`10` vs `ten` / `दस`).
- **System FFmpeg Dependency**: Compressed audio formats (`.mp3`, `.m4a`) require FFmpeg to be installed on the host OS.

---

## 23. Future Improvements

- Add support for larger Whisper checkpoints (`small`, `medium`, `large-v3-turbo`) and `faster-whisper` (CTranslate2) quantization.
- Integrate Indic-specific number-to-word and punctuation normalizers.
- Add speaker/SNR metadata columns and confidence/segment timestamp inspection.
- Support batch audio archive (`.zip`) uploads in the Streamlit dashboard.
