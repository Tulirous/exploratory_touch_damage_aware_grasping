from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from datasets.fragile_episode_dataset import FragileEpisodeDataset
from models.fastwam_fragile import FastWAMFragile


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> FastWAMFragile:
    binary = cfg["outcomes"]["binary"]
    regression = cfg["outcomes"]["regression"]
    return FastWAMFragile(
        visual_latent_dim=cfg["backbone"]["visual_latent_dim"],
        use_text=bool(cfg["model"].get("use_text", True)),
        text_vocab_size=int(cfg["model"].get("text_vocab_size", 4096)),
        text_max_tokens=int(cfg["model"].get("text_max_tokens", 32)),
        text_latent_dim=int(cfg["model"].get("text_latent_dim", 256)),
        tactile_input_dim=cfg["data"]["tactile_channels"],
        action_dim=cfg["data"]["action_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        tactile_latent_dim=cfg["model"]["tactile_latent_dim"],
        action_latent_dim=cfg["model"]["action_latent_dim"],
        fusion_layers=cfg["model"]["fusion_layers"],
        fusion_heads=cfg["model"]["fusion_heads"],
        num_binary=len(binary),
        num_regression=len(regression),
        dropout=cfg["model"]["dropout"],
    )


def compute_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    binary_logits = outputs["binary_logits"]
    regression = outputs["regression"]

    if binary_logits.ndim == 3:
        # If K candidate actions are provided, train all candidates against the
        # episode-level label for now. Later, prefer per-candidate labels.
        binary_targets = batch["binary_labels"].unsqueeze(1).expand_as(binary_logits)
        regression_targets = batch["regression_labels"].unsqueeze(1).expand_as(regression)
    else:
        binary_targets = batch["binary_labels"]
        regression_targets = batch["regression_labels"]

    loss_binary = F.binary_cross_entropy_with_logits(binary_logits, binary_targets)
    loss_regression = F.mse_loss(regression, regression_targets)
    return loss_binary + loss_regression


def move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fastwam_fragile.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")

    binary = cfg["outcomes"]["binary"]
    regression = cfg["outcomes"]["regression"]
    train_set = FragileEpisodeDataset(cfg["data"]["train_manifest"], binary, regression)
    val_set = FragileEpisodeDataset(cfg["data"]["val_manifest"], binary, regression)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    checkpoint_dir = Path(cfg["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                batch["visual_latent"],
                batch["task_instruction"],
                batch["tactile"],
                batch["candidate_actions"],
            )
            loss = compute_loss(outputs, batch)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = move_batch(batch, device)
                outputs = model(
                    batch["visual_latent"],
                    batch["task_instruction"],
                    batch["tactile"],
                    batch["candidate_actions"],
                )
                val_loss += float(compute_loss(outputs, batch).item())

        train_loss /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
            },
            checkpoint_dir / "latest.pt",
        )


if __name__ == "__main__":
    main()
