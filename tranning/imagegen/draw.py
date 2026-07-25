"""Image generation — from-scratch convolutional VAE (no external AI API).

"製圖": given a folder of example images, learn to draw *new* images in the
same general style (unconditional generation — it does not take a text
prompt like "draw a cat"; that would need paired text/image data, a much
bigger ask than what's available right now). Point it at a real image
folder once one exists; until then this only defines the architecture and
training loop, smoke-tested against synthetic images in test_draw.py.

Architecture: a convolutional encoder maps an image to a Gaussian latent
distribution (mu, logvar), the reparameterization trick samples a latent
vector z from it, and a convolutional decoder reconstructs an image from z.
Training minimizes reconstruction error + KL divergence against a standard
normal prior (beta-weighted, i.e. a beta-VAE). Generating new images is just
sampling z ~ N(0, I) and running it through the decoder.

Why a VAE and not a GAN: a GAN (adversarial generator/discriminator) is the
more common "draw new images" architecture and can produce sharper results,
but it is notoriously unstable to train — generator/discriminator can fall
into mode collapse (always producing near-identical output) exactly like
what happened to sinco's code-model checkpoint earlier in this project
(tranning/chats.py's history). A VAE has no adversarial min-max game, just a
single well-behaved loss to descend, so it reliably converges to *something*
recognisable instead of gambling on GAN training dynamics. The honest
tradeoff: VAE output is characteristically blurrier than a well-trained
GAN/diffusion model — that is an inherent property of pixel-wise
reconstruction loss, not a bug to "just fix" here.

Design choices made to avoid overfitting AND underfitting (same reasoning
style as road_sign_train.py / OCR.py / speech_to_text.py):

  Underfitting:
    - Enough conv channels / a configurable --latent-dim (default 128) so
      the bottleneck isn't strangling capacity before real data arrives.
    - ReduceLROnPlateau on validation loss instead of a fixed LR.

  Overfitting:
    - Dropout in the encoder/decoder conv stack.
    - Weight decay on the optimizer.
    - --beta (KL weight) lets the latent space be regularized harder if the
      dataset is small, at the cost of blurrier reconstructions — a real
      dial to turn once we know the real dataset size, not a fixed guess.
    - Optional horizontal-flip augmentation (--hflip, default on): fine for
      ordinary photos, but turn it OFF for anything with fixed orientation
      semantics (diagrams, text, symbols) the same way road_sign_train.py /
      circuit_diagram_train.py / OCR.py all disable flips for exactly that
      reason — a flipped diagram/character can mean something else entirely.
    - Early stopping: keeps the best-val-loss checkpoint, not the last one.
    - Every epoch prints the train/val loss gap and warns if it's large
      (overfitting) or both losses are barely moving (underfitting).

Usage:
    python draw.py --train-dir path/to/images --epochs 100
    python draw.py --generate 4 --out-dir generated/   # sample from the last checkpoint
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms

IMAGE_SIZE = 64
LATENT_DIM = 128
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "draw_runs"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class ImageFolderDataset(Dataset):
    def __init__(self, image_dir: Path, image_size: int, hflip: bool):
        self.paths = sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not self.paths:
            raise ValueError(f"no images found in {image_dir}")

        ops = [transforms.Resize((image_size, image_size))]
        if hflip:
            ops.append(transforms.RandomHorizontalFlip())
        ops.append(transforms.ToTensor())  # -> [0, 1], shape (3, H, W)
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(image)


class Encoder(nn.Module):
    def __init__(self, latent_dim: int, image_size: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2), nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2), nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2), nn.Dropout2d(0.1),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
        )
        reduced = image_size // 16  # four stride-2 convs
        self.flat_size = 256 * reduced * reduced
        self.reduced = reduced
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, image_size: int):
        super().__init__()
        reduced = image_size // 16
        self.reduced = reduced
        self.fc = nn.Linear(latent_dim, 256 * reduced * reduced)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout2d(0.1),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout2d(0.1),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 256, self.reduced, self.reduced)
        return self.deconv(h)


class VAE(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM, image_size: int = IMAGE_SIZE):
        super().__init__()
        self.encoder = Encoder(latent_dim, image_size)
        self.decoder = Decoder(latent_dim, image_size)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decoder(z), mu, logvar


def vae_loss(recon, target, mu, logvar, beta: float):
    recon_loss = F.binary_cross_entropy(recon, target, reduction="sum") / target.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.size(0)
    return recon_loss + beta * kl, recon_loss, kl


def train(train_dir: Path, out_dir: Path, epochs: int, batch_size: int, latent_dim: int,
          image_size: int, beta: float, lr: float, val_split: float, hflip: bool,
          patience: int = 15, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ImageFolderDataset(train_dir, image_size, hflip)
    n_val = max(1, int(len(dataset) * val_split)) if len(dataset) > 1 else 0
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(42)
    if n_val > 0:
        train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val], generator=generator)
    else:
        train_set, val_set = dataset, dataset  # too few images to split; val just mirrors train

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = VAE(latent_dim, image_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps({"latent_dim": latent_dim, "image_size": image_size}, indent=2)
    )

    best_val = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss, _, _ = vae_loss(recon, batch, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)
        train_loss /= len(train_set)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon, mu, logvar = model(batch)
                loss, _, _ = vae_loss(recon, batch, mu, logvar, beta)
                val_loss += loss.item() * batch.size(0)
        val_loss /= len(val_set)

        scheduler.step(val_loss)

        gap = val_loss - train_loss
        warning = ""
        if train_loss > 0 and gap > 0.3 * abs(train_loss):
            warning = "  [警告：train/val 差距偏大，可能過擬合]"
        elif history and epoch > 5 and abs(history[-1]["train_loss"] - train_loss) < 1e-4:
            warning = "  [警告：loss 幾乎沒在下降，可能欠擬合或學習率太小]"
        print(f"epoch {epoch:3d}  train={train_loss:.2f}  val={val_loss:.2f}{warning}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), out_dir / "vae.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"early stopping：驗證 loss 已經 {patience} 個 epoch 沒有進步")
                break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return model, history


_loaded = None


def _load(out_dir: Path):
    global _loaded
    key = str(out_dir)
    if _loaded is not None and _loaded[0] == key:
        return _loaded[1]

    import json
    config_path, weights_path = Path(out_dir) / "config.json", Path(out_dir) / "vae.pt"
    if not (config_path.exists() and weights_path.exists()):
        return None

    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = VAE(config["latent_dim"], config["image_size"])
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    _loaded = (key, (model, config))
    return model, config


def generate(n: int, out_dir: Path = DEFAULT_OUT_DIR, save_to: Path | None = None) -> list[Path]:
    """Sample `n` new images from the trained decoder.

    Returns [] with a printed message (rather than crashing) if no
    checkpoint has been trained yet at out_dir.
    """
    loaded = _load(Path(out_dir))
    if loaded is None:
        print("模型尚未訓練，請先提供圖片資料夾並執行 `python draw.py --train-dir <folder>` 進行訓練。")
        return []

    model, config = loaded
    save_to = Path(save_to) if save_to else Path(out_dir) / "samples"
    save_to.mkdir(parents=True, exist_ok=True)

    paths = []
    with torch.no_grad():
        z = torch.randn(n, config["latent_dim"])
        images = model.decoder(z)
        for i, image in enumerate(images):
            path = save_to / f"sample_{i:03d}.png"
            transforms.ToPILImage()(image).save(path)
            paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description="Train or sample a from-scratch VAE image generator")
    parser.add_argument("--train-dir", type=Path, help="folder of training images")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--beta", type=float, default=1.0, help="KL weight; higher = more regularized/blurrier")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--hflip", dest="hflip", action="store_true", default=True,
                         help="random horizontal flip augmentation (default on; turn off for oriented content)")
    parser.add_argument("--no-hflip", dest="hflip", action="store_false")
    parser.add_argument("--generate", type=int, default=0, metavar="N",
                         help="skip training; sample N images from --out-dir's checkpoint")
    args = parser.parse_args()

    if args.generate:
        paths = generate(args.generate, out_dir=args.out_dir)
        for p in paths:
            print(f"寫入：{p}")
        return

    if not args.train_dir:
        parser.error("--train-dir is required unless --generate is given")

    train(args.train_dir, args.out_dir, args.epochs, args.batch_size, args.latent_dim,
          args.image_size, args.beta, args.lr, args.val_split, args.hflip)


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    main()
