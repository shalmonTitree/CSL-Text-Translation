import torch


def collate_fn(batch, tokenizer, max_text_len=64):
    """
    Collate function for CSLT batches.
    """

    feats = torch.stack([b["feats"] for b in batch], dim=0)
    feat_mask = torch.stack([b["feat_mask"] for b in batch], dim=0)
    texts = [b["text"] for b in batch]

    tokenized = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_text_len,
        return_tensors="pt",
    )

    labels = tokenized["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "feats": feats,
        "feat_mask": feat_mask,
        "labels": labels,
        "decoder_attention_mask": tokenized["attention_mask"],
    }