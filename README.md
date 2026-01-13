# Gloss-Free Continuous Sign Language to Text Translation

End-to-end pipeline for translating continuous sign language videos directly into natural language text without gloss supervision.

## Features
- MediaPipe Holistic pose extraction
- Transformer bottleneck encoder
- Iterative masking and Conv1D smoothing
- T5-based text decoder
- Evaluated on PHOENIX-2014-T

## Run Training
python scripts/train.py

## Run Evaluation
python scripts/eval_all_metrics.py

Note: Dataset and extracted features are not included.
