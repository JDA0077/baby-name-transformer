import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data_utils import (
    build_vocab,
    encode_name,
    load_names,
    split_names,
)
from model import NameTransformer


DATA_PATH = "data/names.txt"
CHECKPOINT_PATH = "name_transformer.pt"

BATCH_SIZE = 32
BLOCK_SIZE = 20
EMBED_DIM = 128
NUM_LAYERS = 4
NUM_HEADS = 4
DROPOUT = 0.1

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
EPOCHS = 80

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class NameDataset(Dataset):
    def __init__(self, names, stoi, block_size):
        self.samples = []

        for name in names:
            tokens = encode_name(name, stoi)

            if len(tokens) < 2:
                continue

            x = tokens[:-1]
            y = tokens[1:]

            if len(x) > block_size:
                x = x[:block_size]
                y = y[:block_size]

            pad_id = stoi[SPECIAL_PAD]

            x = x + [pad_id] * (block_size - len(x))
            y = y + [pad_id] * (block_size - len(y))

            self.samples.append(
                (
                    torch.tensor(x, dtype=torch.long),
                    torch.tensor(y, dtype=torch.long),
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


SPECIAL_PAD = "_"


def evaluate(model, loader, pad_id):
    model.eval()

    total_loss = 0.0
    batches = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits, _ = model(x)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=pad_id,
            )

            total_loss += loss.item()
            batches += 1

    model.train()

    return total_loss / max(batches, 1)


def main():
    print(f"Using device: {DEVICE}")

    names = load_names(DATA_PATH)

    train_names, val_names = split_names(
        names,
        validation_ratio=0.2,
        seed=42,
    )

    stoi, itos = build_vocab(names)

    # Add padding token.
    if SPECIAL_PAD not in stoi:
        pad_id = len(stoi)
        stoi[SPECIAL_PAD] = pad_id
        itos[pad_id] = SPECIAL_PAD
    else:
        pad_id = stoi[SPECIAL_PAD]

    print(f"Names: {len(names)}")
    print(f"Training names: {len(train_names)}")
    print(f"Validation names: {len(val_names)}")
    print(f"Vocabulary size: {len(stoi)}")

    train_dataset = NameDataset(
        train_names,
        stoi,
        BLOCK_SIZE,
    )

    val_dataset = NameDataset(
        val_names,
        stoi,
        BLOCK_SIZE,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = NameTransformer(
        vocab_size=len(stoi),
        block_size=BLOCK_SIZE,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_train_loss = 0.0
        batches = 0

        # Cosine learning-rate schedule.
        lr_scale = 0.5 * (
            1.0
            + math.cos(
                math.pi * epoch / EPOCHS
            )
        )

        lr = LEARNING_RATE * lr_scale

        for group in optimizer.param_groups:
            group["lr"] = lr

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits, _ = model(x)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=pad_id,
            )

            optimizer.zero_grad(set_to_none=True)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            total_train_loss += loss.item()
            batches += 1

        train_loss = total_train_loss / max(batches, 1)
        val_loss = evaluate(
            model,
            val_loader,
            pad_id,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"lr {lr:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint = {
                "model_state": model.state_dict(),
                "stoi": stoi,
                "itos": itos,
                "block_size": BLOCK_SIZE,
                "embed_dim": EMBED_DIM,
                "num_layers": NUM_LAYERS,
                "num_heads": NUM_HEADS,
                "dropout": DROPOUT,
            }

            torch.save(
                checkpoint,
                CHECKPOINT_PATH,
            )

            print(
                f"  Saved checkpoint -> "
                f"{CHECKPOINT_PATH}"
            )


if __name__ == "__main__":
    main()
