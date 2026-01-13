import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from tqdm import tqdm
import pandas as pd

from src.dataset import PhoenixCSLTDataset
from src.collate import collate_fn
from src.model import CSLTBottleneckModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = T5Tokenizer.from_pretrained("t5-small")

test_ds = PhoenixCSLTDataset("data/manifests/test_manifest.csv", 200)
test_loader = DataLoader(
    test_ds,
    batch_size=6,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, tokenizer),
)

encoder = CSLTBottleneckModel().to(DEVICE)
t5 = T5ForConditionalGeneration.from_pretrained("t5-small").to(DEVICE)

ckpt = torch.load("checkpoints/epoch_30.pt", map_location=DEVICE)
encoder.load_state_dict(ckpt["encoder"])
t5.load_state_dict(ckpt["t5"])

encoder.eval()
t5.eval()

rows = []
idx = 0

with torch.no_grad():
    for batch in tqdm(test_loader):
        feats = batch["feats"].to(DEVICE)
        mask = batch["feat_mask"].to(DEVICE)

        enc_out = encoder(feats, mask, mask_prob=0.0)
        enc_out = BaseModelOutput(last_hidden_state=enc_out)

        ids = t5.generate(
            encoder_outputs=enc_out,
            max_length=48,
            num_beams=2,                 # ✅ smaller beam
            repetition_penalty=1.2,      # ✅ reduces repetition
            early_stopping=True,
        )

        preds = tokenizer.batch_decode(ids, skip_special_tokens=True)

        for p in preds:
            rows.append({
                "prediction": p,
                "ground_truth": test_ds.df.iloc[idx]["translation"],
            })
            idx += 1

pd.DataFrame(rows).to_csv("results_test_predictions.csv", index=False)
print("✅ Saved results_test_predictions.csv")
