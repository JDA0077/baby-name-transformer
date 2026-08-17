import random

SPECIAL_START = "<"
SPECIAL_END = ">"

def load_names(path):
    with open(path, "r", encoding="utf-8") as f:
        return [
            line.strip().lower()
            for line in f
            if line.strip()
        ]

def build_vocab(names):
    chars = sorted(set("".join(names)))
    chars = [SPECIAL_START, SPECIAL_END] + chars

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    return stoi, itos

def encode_name(name, stoi):
    tokens = [stoi[SPECIAL_START]]
    tokens.extend(stoi[ch] for ch in name)
    tokens.append(stoi[SPECIAL_END])
    return tokens

def split_names(names, validation_ratio=0.2, seed=42):
    names = names.copy()
    random.Random(seed).shuffle(names)

    split = int(len(names) * (1.0 - validation_ratio))

    return names[:split], names[split:]
