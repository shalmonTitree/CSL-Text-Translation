Sure 👍
Below is the **complete README.md file only**, clean and properly formatted.
You can **copy-paste this entire content directly into `README.md`** — nothing extra, no explanation.

---

```markdown
# Gloss-Free Continuous Sign Language to Text Translation

This repository implements an **end-to-end deep learning pipeline for translating continuous sign language videos directly into natural language text**, without relying on intermediate gloss annotations.

The system is developed as part of an academic **Master’s Thesis** and focuses on robustness, temporal modelling, and semantic compression.

---

## 🔍 Overview

Continuous Sign Language Translation (CSLT) is challenging due to:

- Long temporal dependencies  
- Co-articulation effects  
- Absence of explicit word boundaries  

This project addresses these challenges by:

- Extracting pose–motion features from video frames  
- Modelling long-range dependencies using a Transformer bottleneck encoder  
- Applying iterative masking and latent smoothing  
- Generating grammatically coherent sentences using a neural text decoder  

The pipeline consists of **seven clearly defined stages**, from raw video input to final evaluation.

---

## ✨ Key Features

- Gloss-free **Continuous Sign Language → Text** translation  
- **MediaPipe Holistic** for pose–motion feature extraction  
- **Transformer bottleneck encoder** for temporal compression  
- **Iterative masking** for robustness  
- **Conv1D-based smoothing** for latent refinement  
- **T5-based text decoder**  
- Standard machine translation evaluation metrics  

---

## 📁 Project Structure

```

csltCode/
│
├── src/                     # Core model components
│   ├── model.py             # Bottleneck Transformer + smoothing
│   ├── dataset.py           # Dataset loader
│   └── collate.py           # Padding and masking logic
│
├── scripts/                 # Executable pipeline stages
│   ├── extract_holistic.py  # Pose feature extraction
│   ├── build_manifest.py    # Manifest generation
│   ├── train.py             # Model training
│   ├── eval_all_metrics.py  # Evaluation metrics
│   └── thesis_report.py     # Thesis tables and plots
│
├── data/                    # (Not included in GitHub)
│   ├── raw/                 # PHOENIX-2014-T dataset
│   ├── features/            # Extracted pose features (.npy)
│   └── manifests/           # Train / Dev / Test CSV files
│
├── checkpoints/             # Saved model checkpoints
├── results/                 # Predictions, metrics, visualisations
└── README.md

```

---

## 📦 Dataset

This project uses the **PHOENIX-2014-T** dataset for continuous sign language translation.

⚠️ **Note**  
Due to size constraints, the dataset, extracted features, checkpoints, and results are **not included** in this repository.

### Expected Local Dataset Structure

```

data/raw/PHOENIX-2014-T/

````

---

## 🔄 Pipeline Stages

### Stage 1 — Input Preprocessing

- Organises raw video frames  
- Validates alignment between frames and sentence annotations  
- Ensures consistent and valid samples  

*(Provided by the dataset — no script required)*

---

### Stage 2 — Feature Extraction

- Applies **MediaPipe Holistic** on each frame  
- Extracts **225-dimensional pose–motion features**  
- Produces a temporal feature matrix per video  

#### Run

```bash
python scripts/extract_holistic.py \
  --frames_root data/raw/PHOENIX-2014-T/features/fullFrame-210x260px \
  --out_root data/features \
  --split train
````

Repeat for `dev` and `test`.

#### Output

```
data/features/train/*.npy
data/features/dev/*.npy
data/features/test/*.npy
```

Each feature file has shape:

```
(T, 225)
```

---

### Stage 3 — Manifest Generation

* Links extracted features with sentence annotations
* Produces CSV files used during training

#### Run

```bash
python scripts/build_manifest.py \
  --corpus_csv data/raw/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv \
  --features_root data/features/train \
  --out data/manifests/train_manifest.csv
```

Repeat for `dev` and `test`.

---

### Stage 4–6 — Model Training

Includes:

* Transformer bottleneck encoder
* Iterative masking of latent tokens
* Conv1D-based smoothing
* T5 text decoder

#### Training Configuration

* Batch size: `6`
* Learning rate: `3e-5`
* Optimizer: `AdamW`
* Mask probability: `0.15`
* Epochs: `20–30+`

#### Run

```bash
python scripts/train.py
```

#### Outputs

```
checkpoints/
├── epoch_1.pt
├── epoch_2.pt
└── ...
```

---

### Stage 7 — Evaluation

* Generates translations on the test set
* Computes standard translation metrics

#### Run

```bash
python scripts/eval_all_metrics.py
```

#### Output

```
results/results_test_predictions.csv
```

---

## 📊 Evaluation Metrics

* BLEU
* chrF
* METEOR
* Word Error Rate (WER)
* Sentence-level Accuracy

---

## 📈 Optional — Thesis Reports & Visualisation

Generates tables, plots, and summaries for thesis writing.

#### Run

```bash
python scripts/thesis_report.py
```

#### Outputs

```
results/plots/
results/tables/
results/summary.txt
```

---

## ▶️ End-to-End Execution Order

1. Prepare PHOENIX-2014-T dataset
2. Run `extract_holistic.py`
3. Run `build_manifest.py`
4. Run `train.py`
5. Run `eval_all_metrics.py`
6. *(Optional)* Run `thesis_report.py`

---

## 🧪 Notes

* `data/`, `checkpoints/`, and `results/` are ignored in GitHub
* Only source code and documentation are version-controlled
* Designed for academic research and reproducibility

---

## 📄 License

For academic and research use only.

---

## ✍️ Author

**Shalmon Titre**
Master’s Thesis Project
Gloss-Free Continuous Sign Language Translation
