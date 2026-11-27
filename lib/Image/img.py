"""nano banana — text-to-image generation + image editing + deblurring, from-scratch.

Extends tranning/imagegen/draw.py's unconditional VAE with a text condition, reusing its
Encoder/Decoder conv stacks (subclassed here, not copy-pasted) instead of a from-scratch
diffusion/GAN model — same reasoning draw.py already documents: a VAE has one well-behaved
loss to descend instead of an adversarial min-max game or a heavy multi-stage diffusion
sampler, which stays realistic on an RTX 4060 8GB (Rule 06). Expect draw.py's same honest
tradeoff (blurry, not photoreal), now compounded by a from-scratch character-level text
condition standing in for a real language-vision encoder.

Three capabilities:
  1. generate(prompts)         text only -> new image (condition = prompt, z ~ N(0, I))
  2. edit(image, instruction)  image + text -> edited image (condition = instruction,
                                z = source image's own latent, so "content" comes from the
                                image and "change" comes from the text)
  3. deblur(image)             blurry image -> sharpened image (no text, no VAE bottleneck —
                                see DeblurNet below for why this one is a different
                                architecture from the other two)

generate()/edit() have no paired dataset yet (decided 2026-09-12 via AskUserQuestion,
alongside "put the code directly in lib/img.py" instead of tranning/imagegen/ — see
CLAUDE.md architecture note for that deviation); only smoke-tested against synthetic data
(test_img.py), same "skeleton first" pattern as OCR.py / road_sign_train.py. deblur()
is different: it needs no purpose-built dataset at all, only an ordinary folder of sharp
photos — see DeblurDataset.

Manifests once real generate()/edit() data exists (paths relative to the manifest's folder):
  text-to-image : [{"image": "a.png", "caption": "a red circle"}, ...]
  editing       : [{"source": "before.png", "instruction": "make it blue", "target": "after.png"}, ...]

Usage:
    python img.py --train-t2i captions.json --epochs 100
    python img.py --train-edit edits.json --epochs 100
    python img.py --train-deblur path/to/sharp_photos/ --epochs 100
    python img.py --generate "a red circle" "a blue square"
    python img.py --edit before.png "make it blue"
    python img.py --deblur blurry.jpg
"""

import argparse, json, random, sys
from pathlib import Path

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tranning" / "imagegen"))
import draw

IMAGE_SIZE = draw.IMAGE_SIZE
LATENT_DIM = draw.LATENT_DIM
COND_DIM = 64
DEFAULT_T2I_DIR = Path(__file__).resolve().parent / "img_runs" / "t2i"
DEFAULT_EDIT_DIR = Path(__file__).resolve().parent / "img_runs" / "edit"
DEFAULT_DEBLUR_DIR = Path(__file__).resolve().parent / "img_runs" / "deblur"


def build_vocab(texts: list[str]) -> dict[str, int]:
    """Char-level vocab (same simplicity as chats.py's tokenizer) — 0 is reserved for unknown/pad."""
    return {ch: i + 1 for i, ch in enumerate(sorted({ch for t in texts for ch in t}))}


class CharTextEncoder(nn.Module):
    """Char embedding + mean pool -> one condition vector per text. No unseen-char crash: falls
    back to the unk/pad embedding, same contract as chats.py's own tokenizer."""

    def __init__(self, vocab: dict, cond_dim: int):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(len(vocab) + 1, cond_dim, padding_idx=0)

    def forward(self, texts: list[str], device) -> torch.Tensor:
        vectors = []
        for text in texts:
            ids = torch.tensor([self.vocab.get(ch, 0) for ch in text] or [0], dtype=torch.long, device=device)
            vectors.append(self.embed(ids).mean(dim=0))
        return torch.stack(vectors)


class CondEncoder(draw.Encoder):
    """draw.Encoder's conv stack + condition concatenated in before the mu/logvar heads."""

    def __init__(self, latent_dim: int, image_size: int, cond_dim: int):
        super().__init__(latent_dim, image_size)
        self.fc_mu = nn.Linear(self.flat_size + cond_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size + cond_dim, latent_dim)

    def forward(self, x, cond):
        h = torch.cat([self.conv(x).flatten(1), cond], dim=1)
        return self.fc_mu(h), self.fc_logvar(h)


class CondDecoder(draw.Decoder):
    """draw.Decoder's deconv stack + condition concatenated onto z before the projection."""

    def __init__(self, latent_dim: int, image_size: int, cond_dim: int):
        super().__init__(latent_dim, image_size)
        self.fc = nn.Linear(latent_dim + cond_dim, 256 * self.reduced * self.reduced)

    def forward(self, z, cond):
        h = self.fc(torch.cat([z, cond], dim=1)).view(-1, 256, self.reduced, self.reduced)
        return self.deconv(h)


class TextToImageVAE(nn.Module):
    """CVAE: (image, caption) -> mu/logvar; (z, caption) -> image. Generation samples
    z ~ N(0, I) and conditions on a brand-new prompt instead of an encoded image."""

    def __init__(self, vocab: dict, latent_dim: int = LATENT_DIM, image_size: int = IMAGE_SIZE, cond_dim: int = COND_DIM):
        super().__init__()
        self.vocab, self.latent_dim = vocab, latent_dim
        self.text_encoder = CharTextEncoder(vocab, cond_dim)
        self.encoder = CondEncoder(latent_dim, image_size, cond_dim)
        self.decoder = CondDecoder(latent_dim, image_size, cond_dim)

    def forward(self, x, captions):
        cond = self.text_encoder(captions, x.device)
        mu, logvar = self.encoder(x, cond)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decoder(z, cond), mu, logvar

    @torch.no_grad()
    def generate(self, prompts: list[str], device="cpu"):
        cond = self.text_encoder(prompts, device)
        z = torch.randn(len(prompts), self.latent_dim, device=device)
        return self.decoder(z, cond)


class ImageEditor(nn.Module):
    """Plain (unconditioned) image encoder -> z carries the source image's content;
    the instruction only enters through the decoder's condition, so the same source
    image + a different instruction can decode to a different edit."""

    def __init__(self, vocab: dict, latent_dim: int = LATENT_DIM, image_size: int = IMAGE_SIZE, cond_dim: int = COND_DIM):
        super().__init__()
        self.vocab, self.latent_dim = vocab, latent_dim
        self.text_encoder = CharTextEncoder(vocab, cond_dim)
        self.encoder = draw.Encoder(latent_dim, image_size)
        self.decoder = CondDecoder(latent_dim, image_size, cond_dim)

    def forward(self, source, instructions):
        cond = self.text_encoder(instructions, source.device)
        mu, logvar = self.encoder(source)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decoder(z, cond), mu, logvar

    @torch.no_grad()
    def edit(self, source, instructions: list[str]):
        cond = self.text_encoder(instructions, source.device)
        mu, _ = self.encoder(source)
        return self.decoder(mu, cond)


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1), nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2), nn.Dropout2d(0.1),
        )

    def forward(self, x):
        return self.net(x)


class DeblurNet(nn.Module):
    """Small U-Net with skip connections, predicting a residual (sharp - blurry) added back
    onto the input rather than reconstructing the whole image from scratch — a much easier
    target that trains faster and degrades gracefully toward "no change" instead of noise
    if undertrained. Deliberately NOT built on draw.py's Encoder/Decoder: those compress the
    whole image into a small latent vector (fine for "draw something in this style"), which
    would throw away exactly the fine texture detail a sharpening task needs to recover.
    Skip connections here keep that detail flowing around the bottleneck instead."""

    def __init__(self):
        super().__init__()
        self.enc1 = _ConvBlock(3, 32)
        self.enc2 = _ConvBlock(32, 64, stride=2)
        self.enc3 = _ConvBlock(64, 128, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = _ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = _ConvBlock(64, 32)
        self.out = nn.Conv2d(32, 3, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        residual = torch.tanh(self.out(d1))
        return torch.clamp(x + residual, 0.0, 1.0)


def _image_transform(image_size: int):
    # captions/instructions describe a specific look -> no random flip, same reasoning
    # draw.py gives for any oriented content (road signs, circuit symbols, text).
    return transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])


class CaptionedImageDataset(Dataset):
    """[{"image": ..., "caption": ...}, ...] manifest for text-to-image training."""

    def __init__(self, manifest_path, image_size: int = IMAGE_SIZE):
        base = Path(manifest_path).parent
        entries = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.items = [(base / e["image"], e["caption"]) for e in entries]
        self.transform = _image_transform(image_size)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, caption = self.items[idx]
        image = self.transform(Image.open(path).convert("RGB"))
        return (image, caption), image


class EditImageDataset(Dataset):
    """[{"source": ..., "instruction": ..., "target": ...}, ...] manifest for editing training."""

    def __init__(self, manifest_path, image_size: int = IMAGE_SIZE):
        base = Path(manifest_path).parent
        entries = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.items = [(base / e["source"], e["instruction"], base / e["target"]) for e in entries]
        self.transform = _image_transform(image_size)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        source_path, instruction, target_path = self.items[idx]
        source = self.transform(Image.open(source_path).convert("RGB"))
        target = self.transform(Image.open(target_path).convert("RGB"))
        return (source, instruction), target


class DeblurDataset(Dataset):
    """Two modes:
    - `source` is a folder of ordinary sharp photos: synthesizes a blurry input by applying a
      random-strength Gaussian blur on the fly each time an item is read, so no purpose-built
      "blurry/sharp" dataset is required, and even a handful of photos yields varied pairs
      across epochs (different random sigma each read).
    - `source` is a manifest.json ([{"blurry": ..., "sharp": ...}, ...]): use real paired
      blurry/sharp photos instead of synthetic blur, for whenever real ones are available.
    """

    def __init__(self, source, image_size: int = IMAGE_SIZE, blur_sigma=(2.0, 6.0)):
        source = Path(source)
        self.transform = _image_transform(image_size)
        self.blur_sigma = blur_sigma
        if source.suffix.lower() == ".json":
            base = source.parent
            entries = json.loads(source.read_text(encoding="utf-8"))
            self.pairs = [(base / e["blurry"], base / e["sharp"]) for e in entries]
            self.synthetic = False
        else:
            self.pairs = sorted(p for p in source.iterdir() if p.suffix.lower() in draw.IMAGE_EXTENSIONS)
            if not self.pairs:
                raise ValueError(f"no images found in {source}")
            self.synthetic = True

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if self.synthetic:
            sharp: torch.Tensor = self.transform(Image.open(self.pairs[idx]).convert("RGB"))  # type: ignore[assignment]
            sigma = random.uniform(*self.blur_sigma)
            blurry = TF.gaussian_blur(sharp, kernel_size=[9, 9], sigma=[sigma, sigma])
        else:
            blurry_path, sharp_path = self.pairs[idx]
            blurry = self.transform(Image.open(blurry_path).convert("RGB"))  # type: ignore[assignment]
            sharp = self.transform(Image.open(sharp_path).convert("RGB"))  # type: ignore[assignment]
        return blurry, sharp


def _split(dataset, val_split: float):
    n_val = max(1, int(len(dataset) * val_split)) if len(dataset) > 1 else 0
    if not n_val:
        return dataset, dataset
    generator = torch.Generator().manual_seed(42)
    return torch.utils.data.random_split(dataset, [len(dataset) - n_val, n_val], generator=generator)


def _vae_loss_fn(beta: float):
    def loss_fn(model, inputs, target):
        recon, mu, logvar = model(*inputs)
        loss, _, _ = draw.vae_loss(recon, target, mu, logvar, beta)
        return loss
    return loss_fn


def _deblur_loss_fn(model, inputs, target):
    return F.l1_loss(model(*inputs), target)


def _run_epoch(model, loader, optimizer, loss_fn, device, train_mode: bool):
    model.train(train_mode)
    total, count = 0.0, 0
    with torch.set_grad_enabled(train_mode):
        for inputs, target in loader:
            if torch.is_tensor(inputs):
                inputs = (inputs,)
            inputs = tuple(x.to(device) if torch.is_tensor(x) else x for x in inputs)
            target = target.to(device)
            if train_mode:
                optimizer.zero_grad()
            loss = loss_fn(model, inputs, target)
            if train_mode:
                loss.backward()
                optimizer.step()
            total += loss.item() * target.size(0)
            count += target.size(0)
    return total / count


def _train(model, train_set, val_set, out_dir: Path, epochs: int, batch_size: int, loss_fn,
           lr: float, patience: int, device):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val, no_improve, history = float("inf"), 0, []
    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, optimizer, loss_fn, device, True)
        val_loss = _run_epoch(model, val_loader, optimizer, loss_fn, device, False)
        scheduler.step(val_loss)

        gap, warning = val_loss - train_loss, ""
        if train_loss > 0 and gap > 0.3 * abs(train_loss):
            warning = "  [警告：train/val 差距偏大，可能過擬合]"
        elif history and epoch > 5 and abs(history[-1]["train_loss"] - train_loss) < 1e-4:
            warning = "  [警告：loss 幾乎沒在下降，可能欠擬合或學習率太小]"
        print(f"epoch {epoch:3d}  train={train_loss:.2f}  val={val_loss:.2f}{warning}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val, no_improve = val_loss, 0
            torch.save(model.state_dict(), out_dir / "model.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"early stopping：驗證 loss 已經 {patience} 個 epoch 沒有進步")
                break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return history


def train_text_to_image(manifest_path, out_dir=DEFAULT_T2I_DIR, epochs=100, batch_size=16,
                         latent_dim=LATENT_DIM, image_size=IMAGE_SIZE, cond_dim=COND_DIM,
                         beta=1.0, lr=1e-3, val_split=0.1, patience=15, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CaptionedImageDataset(manifest_path, image_size)
    vocab = build_vocab([caption for _, caption in dataset.items])
    train_set, val_set = _split(dataset, val_split)

    model = TextToImageVAE(vocab, latent_dim, image_size, cond_dim).to(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        {"latent_dim": latent_dim, "image_size": image_size, "cond_dim": cond_dim, "vocab": vocab}, indent=2))

    history = _train(model, train_set, val_set, out_dir, epochs, batch_size, _vae_loss_fn(beta), lr, patience, device)
    return model, history


def train_editor(manifest_path, out_dir=DEFAULT_EDIT_DIR, epochs=100, batch_size=16,
                  latent_dim=LATENT_DIM, image_size=IMAGE_SIZE, cond_dim=COND_DIM,
                  beta=1.0, lr=1e-3, val_split=0.1, patience=15, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = EditImageDataset(manifest_path, image_size)
    vocab = build_vocab([instruction for _, instruction, _ in dataset.items])
    train_set, val_set = _split(dataset, val_split)

    model = ImageEditor(vocab, latent_dim, image_size, cond_dim).to(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        {"latent_dim": latent_dim, "image_size": image_size, "cond_dim": cond_dim, "vocab": vocab}, indent=2))

    history = _train(model, train_set, val_set, out_dir, epochs, batch_size, _vae_loss_fn(beta), lr, patience, device)
    return model, history


def train_deblur(source, out_dir=DEFAULT_DEBLUR_DIR, epochs=100, batch_size=16, image_size=IMAGE_SIZE,
                  lr=1e-3, val_split=0.1, patience=15, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DeblurDataset(source, image_size)
    train_set, val_set = _split(dataset, val_split)

    model = DeblurNet().to(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({"image_size": image_size}, indent=2))

    history = _train(model, train_set, val_set, out_dir, epochs, batch_size, _deblur_loss_fn, lr, patience, device)
    return model, history


_cache: dict = {}


def _load(out_dir, cache_key: str, build_fn):
    key = (str(out_dir), cache_key)
    if key in _cache:
        return _cache[key]
    config_path, weights_path = Path(out_dir) / "config.json", Path(out_dir) / "model.pt"
    if not (config_path.exists() and weights_path.exists()):
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = build_fn(config)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    _cache[key] = (model, config)
    return _cache[key]


def generate(prompts: list[str], out_dir=DEFAULT_T2I_DIR, save_to=None) -> list[Path]:
    """Sample images from text prompts. [] + a printed message if no checkpoint yet."""
    loaded = _load(out_dir, "t2i", lambda c: TextToImageVAE(c["vocab"], c["latent_dim"], c["image_size"], c["cond_dim"]))
    if loaded is None:
        print("文字生圖模型尚未訓練，請先提供圖片+文字說明的 manifest 並執行 --train-t2i 進行訓練。")
        return []
    model, _ = loaded
    save_to = Path(save_to) if save_to else Path(out_dir) / "samples"
    save_to.mkdir(parents=True, exist_ok=True)

    paths = []
    for i, image in enumerate(model.generate(prompts)):
        path = save_to / f"sample_{i:03d}.png"
        transforms.ToPILImage()(image).save(path)
        paths.append(path)
    return paths


def edit(image_path, instruction: str, out_dir=DEFAULT_EDIT_DIR, save_to=None) -> Path | None:
    """Apply a text instruction to one image. None + a printed message if no checkpoint yet."""
    loaded = _load(out_dir, "editor", lambda c: ImageEditor(c["vocab"], c["latent_dim"], c["image_size"], c["cond_dim"]))
    if loaded is None:
        print("修圖模型尚未訓練，請先提供 修改前/指令/修改後 的 manifest 並執行 --train-edit 進行訓練。")
        return None
    model, config = loaded
    transform = _image_transform(config["image_size"])
    image_tensor: torch.Tensor = transform(Image.open(image_path).convert("RGB"))  # type: ignore[assignment]
    source = image_tensor.unsqueeze(0)

    result = model.edit(source, [instruction])[0]
    save_to = Path(save_to) if save_to else Path(out_dir) / "edited.png"
    save_to.parent.mkdir(parents=True, exist_ok=True)
    transforms.ToPILImage()(result).save(save_to)
    return save_to


def deblur(image_path, out_dir=DEFAULT_DEBLUR_DIR, save_to=None) -> Path | None:
    """Sharpen one blurry image. None + a printed message if no checkpoint yet."""
    loaded = _load(out_dir, "deblur", lambda c: DeblurNet())
    if loaded is None:
        print("模糊變清晰模型尚未訓練，請先提供一個清晰圖片資料夾並執行 --train-deblur 進行訓練"
              "（會自動合成模糊/清晰配對，不需要你自己準備模糊圖）。")
        return None
    model, config = loaded
    transform = _image_transform(config["image_size"])
    image_tensor: torch.Tensor = transform(Image.open(image_path).convert("RGB"))  # type: ignore[assignment]

    with torch.no_grad():
        result = model(image_tensor.unsqueeze(0))[0]
    save_to = Path(save_to) if save_to else Path(out_dir) / "deblurred.png"
    save_to.parent.mkdir(parents=True, exist_ok=True)
    transforms.ToPILImage()(result).save(save_to)
    return save_to


def main():
    parser = argparse.ArgumentParser(description="Train/run text-to-image generation, image editing and deblurring (from-scratch)")
    parser.add_argument("--train-t2i", type=Path, help="captioned-image manifest to train the text-to-image model")
    parser.add_argument("--train-edit", type=Path, help="source/instruction/target manifest to train the editing model")
    parser.add_argument("--train-deblur", type=Path, help="folder of sharp photos (or a blurry/sharp manifest) to train the deblurring model")
    parser.add_argument("--generate", nargs="+", metavar="PROMPT", help="generate images from text prompts")
    parser.add_argument("--edit", nargs=2, metavar=("IMAGE", "INSTRUCTION"), help="edit one image with a text instruction")
    parser.add_argument("--deblur", metavar="IMAGE", help="sharpen one blurry image")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if args.generate:
        for path in generate(args.generate, out_dir=args.out_dir or DEFAULT_T2I_DIR):
            print(f"寫入：{path}")
    elif args.edit:
        path = edit(args.edit[0], args.edit[1], out_dir=args.out_dir or DEFAULT_EDIT_DIR)
        if path:
            print(f"寫入：{path}")
    elif args.deblur:
        path = deblur(args.deblur, out_dir=args.out_dir or DEFAULT_DEBLUR_DIR)
        if path:
            print(f"寫入：{path}")
    elif args.train_t2i:
        train_text_to_image(args.train_t2i, out_dir=args.out_dir or DEFAULT_T2I_DIR, epochs=args.epochs, batch_size=args.batch_size)
    elif args.train_edit:
        train_editor(args.train_edit, out_dir=args.out_dir or DEFAULT_EDIT_DIR, epochs=args.epochs, batch_size=args.batch_size)
    elif args.train_deblur:
        train_deblur(args.train_deblur, out_dir=args.out_dir or DEFAULT_DEBLUR_DIR, epochs=args.epochs, batch_size=args.batch_size)
    else:
        parser.error("need one of --train-t2i / --train-edit / --train-deblur / --generate / --edit / --deblur")


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    main()
