import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# CONFIG
# =========================
RESULTS_DIR = "results"
TABLE_DIR = os.path.join(RESULTS_DIR, "tables")
GRAPH_DIR = os.path.join(RESULTS_DIR, "graphs")

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

# Final metrics (LOCKED)
METRICS = {
    "BLEU": 7.02,
    "chrF": 26.34,
    "WER": 0.991,
    "Sentence Accuracy": 1.25,
    "METEOR": 0.167,
}

# =========================
# 1️⃣ METRICS TABLE
# =========================
metrics_df = pd.DataFrame(
    [{"Metric": k, "Score": v} for k, v in METRICS.items()]
)
metrics_df.to_csv(f"{TABLE_DIR}/final_metrics.csv", index=False)

# =========================
# 2️⃣ METRIC BAR GRAPH
# =========================
plt.figure()
plt.bar(metrics_df["Metric"], metrics_df["Score"])
plt.title("Final Evaluation Metrics")
plt.ylabel("Score")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(f"{GRAPH_DIR}/metric_bar_chart.png")
plt.close()

# =========================
# 3️⃣ LOSS CURVE (MANUAL COPY)
# =========================
train_loss = [
    3.78, 2.83, 2.60, 2.46, 2.36, 2.29, 2.23, 2.18, 2.13, 2.09,
    2.06, 2.02, 2.00, 1.96, 1.94, 1.92, 1.89, 1.87, 1.85, 1.82,
    1.77, 1.76, 1.75, 1.74, 1.73, 1.73, 1.72, 1.71, 1.70, 1.70,
]

dev_loss = [
    2.80, 2.46, 2.28, 2.18, 2.07, 2.03, 1.97, 1.94, 1.89, 1.87,
    1.85, 1.82, 1.78, 1.79, 1.75, 1.73, 1.71, 1.70, 1.69, 1.67,
    1.65, 1.64, 1.65, 1.64, 1.63, 1.63, 1.62, 1.63, 1.62, 1.62,
]

plt.figure()
plt.plot(train_loss, label="Train Loss")
plt.plot(dev_loss, label="Dev Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.tight_layout()
plt.savefig(f"{GRAPH_DIR}/loss_curve.png")
plt.close()

# =========================
# 4️⃣ SENTENCE LENGTH ANALYSIS
# =========================
preds = pd.read_csv("results_test_predictions.csv")

pred_lens = preds["prediction"].apply(lambda x: len(str(x).split()))
ref_lens  = preds["ground_truth"].apply(lambda x: len(str(x).split()))

len_df = pd.DataFrame({
    "prediction_length": pred_lens,
    "reference_length": ref_lens
})
len_df.to_csv(f"{TABLE_DIR}/token_length_stats.csv", index=False)

plt.figure()
plt.hist(ref_lens, bins=30, alpha=0.5, label="Reference")
plt.hist(pred_lens, bins=30, alpha=0.5, label="Prediction")
plt.legend()
plt.title("Sentence Length Distribution")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{GRAPH_DIR}/sentence_length_distribution.png")
plt.close()

# =========================
# 5️⃣ PRED VS REF LENGTH
# =========================
plt.figure()
plt.scatter(ref_lens, pred_lens, alpha=0.5)
plt.xlabel("Reference Length")
plt.ylabel("Prediction Length")
plt.title("Prediction vs Reference Length")
plt.tight_layout()
plt.savefig(f"{GRAPH_DIR}/prediction_vs_reference_length.png")
plt.close()

# =========================
# 6️⃣ EXAMPLE PREDICTIONS
# =========================
preds.sample(20).to_csv(
    f"{TABLE_DIR}/example_predictions.csv", index=False
)

# =========================
# 7️⃣ SUMMARY TEXT
# =========================
with open(f"{RESULTS_DIR}/summary.txt", "w", encoding="utf-8") as f:
    f.write("FINAL CSL -> TEXT RESULTS\n")
    f.write("========================\n\n")
    for k, v in METRICS.items():
        f.write(f"{k}: {v}\n")

print("✅ ALL tables and graphs generated successfully.")
