import torch
import torch.nn.functional as F

from model import NameTransformer

CHECKPOINT_PATH = "name_transformer.pt"
DEVICE = "cpu"


def generate_name(model, stoi, itos, block_size, temperature=0.8):
    start_id = stoi["<"]
    end_id = stoi[">"]
    pad_id = stoi["_"]

    tokens = [start_id]

    for _ in range(block_size - 1):
        x = torch.tensor(
            [tokens],
            dtype=torch.long,
            device=DEVICE,
        )

        with torch.no_grad():
            logits, _ = model(x)

        logits = logits[0, -1] / temperature

        # Never generate padding or another start token.
        logits[pad_id] = float("-inf")
        logits[start_id] = float("-inf")

        probs = F.softmax(logits, dim=-1)

        next_id = torch.multinomial(
            probs,
            num_samples=1,
        ).item()

        if next_id == end_id:
            break

        tokens.append(next_id)

    chars = [
        itos[i]
        for i in tokens[1:]
        if i not in (start_id, end_id, pad_id)
    ]

    return "".join(chars)


def main():
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]

    model = NameTransformer(
        vocab_size=len(stoi),
        block_size=checkpoint["block_size"],
        embed_dim=checkpoint["embed_dim"],
        num_layers=checkpoint["num_layers"],
        num_heads=checkpoint["num_heads"],
        dropout=checkpoint["dropout"],
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    print("\nGenerated names")
    print("----------------")

    generated = set()

    while len(generated) < 20:
        name = generate_name(
            model,
            stoi,
            itos,
            checkpoint["block_size"],
            temperature=0.9,
        )

        if len(name) >= 3:
            generated.add(name.capitalize())

    for i, name in enumerate(sorted(generated), 1):
        print(f"{i:2d}. {name}")


if __name__ == "__main__":
    main()
