import pandas as pd
from jiwer import wer
import sacrebleu
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score


CSV_PATH = "results_test_predictions.csv"


def tokenize(s):
    return s.strip().split()


def main():
    df = pd.read_csv(CSV_PATH)

    predictions = df["prediction"].astype(str).tolist()
    references_text = df["ground_truth"].astype(str).tolist()

    # --------------------
    # BLEU
    # --------------------
    hypotheses = [tokenize(p) for p in predictions]
    references = [[tokenize(r)] for r in references_text]
    bleu = corpus_bleu(references, hypotheses) * 100

    # --------------------
    # WER
    # --------------------
    wers = [
        wer(r, p)
        for r, p in zip(references_text, predictions)
    ]
    avg_wer = sum(wers) / len(wers)

    # --------------------
    # Sentence Accuracy
    # --------------------
    sent_acc = sum(
        r.strip() == p.strip()
        for r, p in zip(references_text, predictions)
    ) / len(predictions)

    # --------------------
    # METEOR (FIXED ✅)
    # --------------------
    meteor_scores = [
        meteor_score([tokenize(r)], tokenize(p))
        for r, p in zip(references_text, predictions)
    ]
    avg_meteor = sum(meteor_scores) / len(meteor_scores)

    # --------------------
    # chrF / chrF++
    # --------------------
    chrf = sacrebleu.corpus_chrf(predictions, [references_text]).score

    # --------------------
    # PRINT RESULTS
    # --------------------
    print("\n========== CSL → TEXT EVALUATION ==========")
    print(f"Total samples       : {len(df)}")
    print("-------------------------------------------")
    print(f"BLEU                : {bleu:.2f}")
    print(f"chrF                : {chrf:.2f}")
    print(f"WER                 : {avg_wer:.3f}")
    print(f"Sentence Accuracy   : {sent_acc*100:.2f}%")
    print(f"METEOR              : {avg_meteor:.3f}")
    print("===========================================\n")


if __name__ == "__main__":
    main()
