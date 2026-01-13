# Gloss-Free Continuous Sign Language to Text Translation

This repository implements an **end-to-end deep learning pipeline for translating continuous sign language videos directly into natural language text**, without relying on intermediate gloss annotations. The system is designed as part of an academic Master’s Thesis and focuses on robustness, temporal modelling, and semantic compression.

---

## 🔍 Overview

Continuous Sign Language Translation (CSLT) is a challenging problem due to long temporal dependencies, co-articulation effects, and lack of explicit word boundaries. This project addresses these challenges by:

- Extracting pose–motion features from video frames
- Modelling long-range temporal dependencies using a Transformer-based bottleneck encoder
- Applying iterative masking and latent smoothing for robustness
- Generating grammatically coherent sentences using a neural text decoder

The complete pipeline consists of **seven clearly defined stages**, from raw video input to final evaluation.

---

## ✨ Key Features

- Gloss-free **CSL → Text** translation
- **MediaPipe Holistic** for pose–motion feature extraction
- **Transformer bottleneck encoder** for temporal compression
- **Iterative masking** to improve robustness
- **Conv1D-based smoothing** for latent refinement
- **T5-based text decoder**
- Evaluation using standard machine translation metrics

---

## 📁 Project Structure



csltCode/
│
├── src/ # Core model components
│ ├── model.py # Bottleneck Transformer + smoothing
│ ├── dataset.py # Dataset loader
│ └── collate.py # Padding and masking logic
│
├── scripts/ # Executable pipeline stages
│ ├── extract_holistic.py # Pose feature extraction
│ ├── build_manifest.py # Manifest generation
│ ├── train.py # Model training
│ ├── eval_all_metrics.py # Evaluation metrics
│ └── thesis_report.py # Thesis tables and plots
│
├── data/ # (Not included in GitHub)
│ ├── raw/ # PHOENIX-2014-T dataset
│ ├── features/ # Extracted pose features (.npy)
│ └── manifests/ # Train / Dev / Test CSV files
│
├── checkpoints/ # Saved model checkpoints
├── results/ # Predictions, metrics, visualisations
└── README.md


---

## 📦 Dataset

This project uses the **PHOENIX-2014-T** dataset for continuous sign language translation.

⚠️ **Note:**  
Due to size constraints, the dataset, extracted features, checkpoints, and results are **not included** in this repository.

Expected local dataset structure:


---

## 📦 Dataset

This project uses the **PHOENIX-2014-T** dataset for continuous sign language translation.

⚠️ **Note:**  
Due to size constraints, the dataset, extracted features, checkpoints, and results are **not included** in this repository.

Expected local dataset structure:
data/raw/PHOENIX-2014-T/


---

## 🔄 Pipeline Stages

### Stage 1 — Input Preprocessing
- Organises raw video frames
- Validates alignment between frames and sentence annotations
- Ensures consistent and valid samples

(Provided by the dataset; no script required)

---

### Stage 2 — Feature Extraction
- Applies **MediaPipe Holistic** on each frame
- Extracts 225-dimensional pose–motion descriptors
- Produces a temporal feature matrix per sample

Run:
```bash
python scripts/extract_holistic.py \
  --frames_root data/raw/PHOENIX-2014-T/features/fullFrame-210x260px \
  --out_root data/features \
  --split train


Repeat for dev and test.

Output:

data/features/train/*.npy
data/features/dev/*.npy
data/features/test/*.npy


Each feature file has shape:

(
𝑇
,
225
)
(T,225)
Stage 3 — Manifest Generation

Links extracted features with text annotations

Produces CSV files used by the training pipeline

Run:

python scripts/build_manifest.py \
  --corpus_csv data/raw/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv \
  --features_root data/features/train \
  --out data/manifests/train_manifest.csv


Repeat for dev and test splits.

Stage 4–6 — Model Training

Includes:

Transformer-based bottleneck encoder

Iterative masking of latent tokens

Conv1D-based smoothing

T5 text decoder

Training configuration:

Batch size: 6

Learning rate: 3e-5

Optimizer: AdamW

Mask probability: 0.15

Epochs: 20–30+

Run training:

python scripts/train.py


Outputs:

checkpoints/epoch_1.pt
checkpoints/epoch_2.pt
...

Stage 7 — Evaluation

Generates translations on the test set

Computes standard translation metrics

Run:

python scripts/eval_all_metrics.py


Output:

results/results_test_predictions.csv

Optional — Thesis Reports & Visualisation

Generates tables, plots, and summaries for thesis writing

Run:

python scripts/thesis_report.py


Outputs:

results/plots/
results/tables/
results/summary.txt

▶️ End-to-End Execution Order
1. Prepare PHOENIX-2014-T dataset
2. Run extract_holistic.py
3. Run build_manifest.py
4. Run train.py
5. Run eval_all_metrics.py
6. (Optional) Run thesis_report.py

📊 Evaluation Metrics

BLEU

chrF

METEOR

Word Error Rate (WER)

Sentence-level Accuracy

These metrics provide complementary insights into lexical overlap, semantic similarity, and word-level alignment.

🧪 Notes

data/, checkpoints/, and results/ are ignored in GitHub

Only source code and documentation are version-controlled

Designed for academic research and reproducibility

📄 License

For academic and research use only.

✍️ Author

Shalmon Titre
Master’s Thesis Project
Gloss-Free Continuous Sign Language Translation
