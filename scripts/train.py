import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import AdamW
from src.dataset import PhoenixCSLTDataset
from src.collate import collate_fn
from src.model import CSLTBottleneckModel
from tqdm import tqdm
import os


# =======================
# CONFIG
# =======================
TRAIN_MANIFEST = "data/manifests/train_manifest.csv"
DEV_MANIFEST   = "data/manifests/dev_manifest.csv"

BATCH_SIZE = 6
MAX_FRAMES = 200

START_EPOCH = 21       # ✅ resume from epoch 21
END_EPOCH   = 30       # ✅ train until 30

LR = 1e-5              # ✅ LOWER LR for fine-tuning
MASK_PROB = 0.15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# =======================
# LOAD DATA
# =======================
tokenizer = T5Tokenizer.from_pretrained("t5-small")

train_ds = PhoenixCSLTDataset(TRAIN_MANIFEST, max_frames=MAX_FRAMES)
dev_ds   = PhoenixCSLTDataset(DEV_MANIFEST,   max_frames=MAX_FRAMES)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, tokenizer),
)

dev_loader = DataLoader(
    dev_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, tokenizer),
)


# =======================
# LOAD MODELS
# =======================
encoder = CSLTBottleneckModel().to(DEVICE)
t5 = T5ForConditionalGeneration.from_pretrained("t5-small").to(DEVICE)

# ✅ RESUME FROM EPOCH 20
ckpt = torch.load("checkpoints/epoch_20.pt", map_location=DEVICE)
encoder.load_state_dict(ckpt["encoder"])
t5.load_state_dict(ckpt["t5"])
print("✅ Resumed from checkpoint: epoch_20.pt")

optimizer = AdamW(
    list(encoder.parameters()) + list(t5.parameters()),
    lr=LR
)


# =======================
# TRAIN / VALIDATE
# =======================
def run_epoch(loader, training=True):
    total_loss = 0.0

    if training:
        encoder.train()
        t5.train()
    else:
        encoder.eval()
        t5.eval()

    for batch in tqdm(loader):
        feats = batch["feats"].to(DEVICE)
        feat_mask = batch["feat_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        with torch.set_grad_enabled(training):
            enc_out = encoder(
                feats, feat_mask, mask_prob=(MASK_PROB if training else 0.0)
            )

            outputs = t5(
                encoder_outputs=(enc_out,),
                labels=labels
            )

            loss = outputs.loss

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# =======================
# MAIN LOOP (21 → 30)
# =======================
for epoch in range(START_EPOCH, END_EPOCH + 1):
    print(f"\n===== EPOCH {epoch}/30 =====")

    train_loss = run_epoch(train_loader, training=True)
    dev_loss   = run_epoch(dev_loader,   training=False)

    print(f"Train loss: {train_loss:.4f}")
    print(f"Dev   loss: {dev_loss:.4f}")

    torch.save(
        {
            "encoder": encoder.state_dict(),
            "t5": t5.state_dict(),
            "epoch": epoch,
        },
        f"{CHECKPOINT_DIR}/epoch_{epoch}.pt",
    )

print("\n✅ Fine-tuning complete (epochs 21–30).")
