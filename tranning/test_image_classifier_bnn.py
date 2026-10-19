"""Pipeline smoke test for image_classifier_bnn.py.

Does NOT claim any recognition accuracy — there is no real dataset yet.
Only proves the training loop (dataloaders, model, optimizer, scheduler,
early stopping, checkpoint + classes.json/config.json output) and the
Bayesian (MC Dropout) classify_image() inference path run end-to-end
without crashing, using a tiny synthetic image dataset produced via
data_split.py (same fixture style as test_road_sign_train.py).
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from data_split import split_dataset
from image_classifier_bnn import classify_image, train


def _make_synthetic_dataset(root: Path):
    classes = {
        "cat": (200, 30, 30),
        "dog": (30, 30, 200),
    }
    for name, color in classes.items():
        class_dir = root / name
        class_dir.mkdir(parents=True)
        for i in range(12):
            arr = np.full((64, 64, 3), color, dtype=np.uint8)
            Image.fromarray(arr).save(class_dir / f"{name}_{i}.jpg")


def test_training_loop_runs_end_to_end(tmp_path):
    source = tmp_path / "source"
    split_dir = tmp_path / "split"
    out_dir = tmp_path / "runs"

    _make_synthetic_dataset(source)
    split_dataset(source, split_dir, train=0.7, val=0.15, test=0.15, seed=1)

    model, classes, history = train(
        data_dir=split_dir, out_dir=out_dir, epochs=2, batch_size=4, lr=1e-3,
        patience=5, image_size=96, freeze_backbone=True, dropout=0.3, weight_decay=1e-4,
    )

    assert classes == sorted(["cat", "dog"])
    assert len(history) == 2
    assert (out_dir / "best_model.pt").exists()
    assert json.loads((out_dir / "classes.json").read_text()) == classes
    assert json.loads((out_dir / "config.json").read_text())["dropout"] == 0.3
    assert all(0.0 <= h["val_acc"] <= 1.0 for h in history)


def test_classify_image_after_training_returns_bayesian_stats(tmp_path):
    source = tmp_path / "source"
    split_dir = tmp_path / "split"
    out_dir = tmp_path / "runs"

    _make_synthetic_dataset(source)
    split_dataset(source, split_dir, train=0.7, val=0.15, test=0.15, seed=1)
    train(
        data_dir=split_dir, out_dir=out_dir, epochs=1, batch_size=4, lr=1e-3,
        patience=5, image_size=96, freeze_backbone=True, dropout=0.3, weight_decay=1e-4,
    )

    sample_image = next((split_dir / "train" / "cat").glob("*.jpg"))
    label, confidence, entropy = classify_image(sample_image, out_dir=out_dir, mc_samples=5)

    assert label in {"cat", "dog"}
    assert 0.0 <= confidence <= 1.0
    assert entropy >= 0.0


def test_classify_image_without_checkpoint_returns_placeholder(tmp_path):
    label, confidence, entropy = classify_image(
        Path("does_not_matter.jpg"), out_dir=tmp_path / "no_checkpoint_here"
    )
    assert "尚未訓練" in label
    assert confidence == 0.0
    assert entropy == 0.0


def test_grayscale_training_and_inference(tmp_path, monkeypatch):
    import torch
    import image_classifier_bnn as classifier

    # 小型自建模型驗證管線，不下載權重、不測試辨識準確度。
    monkeypatch.setattr(classifier, "build_model", lambda *args, **kwargs: torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 2)))
    for split in ("train", "val"):
        for name, color in (("red", (255, 0, 0)), ("blue", (0, 0, 255))):
            folder = tmp_path / "data" / split / name
            folder.mkdir(parents=True)
            Image.new("RGB", (16, 16), color).save(folder / "sample.png")
    out_dir = tmp_path / "runs"
    classifier.train(tmp_path / "data", out_dir, 1, 2, 1e-3, 2, 16,
                     False, 0.3, 1e-4, hflip=False, device="cpu", grayscale=True)
    assert json.loads((out_dir / "config.json").read_text())["grayscale"] is True
    with Image.open(out_dir / "training_curves.png") as chart:
        assert chart.width >= 1000 and chart.height >= 400
    sample = tmp_path / "data" / "val" / "red" / "sample.png"
    seen = []

    def predict(model, tensor, samples):
        seen.append(tensor)
        return torch.tensor([[0.7, 0.3]]), torch.zeros(1, 2)

    monkeypatch.setattr(classifier, "mc_predict_classification", predict)
    classifier.classify_image(sample, out_dir)
    expected = Image.open(sample).convert("L").getpixel((0, 0)) / 255
    restored = seen[0][0, :, 0, 0] * torch.tensor(classifier.IMAGENET_STD) + torch.tensor(classifier.IMAGENET_MEAN)
    assert torch.allclose(restored, torch.full((3,), expected), atol=1e-6)
    # 舊 checkpoint 沒有此欄位時，仍沿用彩色輸入。
    config = json.loads((out_dir / "config.json").read_text())
    del config["grayscale"]
    (out_dir / "config.json").write_text(json.dumps(config))
    classifier._loaded_models.clear()
    classifier.classify_image(sample, out_dir)
    restored = seen[-1][0, :, 0, 0] * torch.tensor(classifier.IMAGENET_STD) + torch.tensor(classifier.IMAGENET_MEAN)
    assert torch.allclose(restored, torch.tensor([1., 0., 0.]), atol=1e-6)


def test_grayscale_transforms_match_train_and_validation():
    import torch
    import image_classifier_bnn as classifier

    train_tf, eval_tf = classifier.build_transforms(16, hflip=False, grayscale=True)
    for transform in (train_tf, eval_tf):
        tensor = transform(Image.new("RGB", (16, 16), (220, 30, 70)))
        restored = tensor * torch.tensor(classifier.IMAGENET_STD)[:, None, None] + torch.tensor(classifier.IMAGENET_MEAN)[:, None, None]
        assert tensor.shape == (3, 16, 16)
        assert torch.allclose(restored[0], restored[1], atol=1e-6)
        assert torch.allclose(restored[1], restored[2], atol=1e-6)


def test_plot_history_empty_records(tmp_path):
    import pytest
    from image_classifier_bnn import plot_history

    (tmp_path / "history.json").write_text("[]")
    with pytest.raises(ValueError, match="no training epochs"):
        plot_history(tmp_path)


def test_feature_layers_learn_and_export(tmp_path):
    import torch
    import image_classifier_bnn as classifier

    torch.manual_seed(7)
    model = classifier.build_model(2, freeze_backbone=True)
    assert isinstance(model, classifier.FeatureCNN)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    images = torch.zeros(8, 3, 16, 16)
    images[:4, :, :, 6:10] = 1  # 直線
    images[4:, :, 6:10, :] = 1  # 橫線；亮度相同、形狀不同
    labels = torch.tensor([0] * 4 + [1] * 4)
    before = [block[0].weight.detach().clone() for block in model.features]
    losses = []
    for _ in range(15):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(images), labels)
        loss.backward()
        assert all(block[0].weight.grad.abs().sum() > 0 for block in model.features)
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    assert all(not torch.equal(old, block[0].weight) for old, block in zip(before, model.features))
    torch.save(model.state_dict(), tmp_path / "best_model.pt")
    (tmp_path / "classes.json").write_text('["vertical", "horizontal"]')
    (tmp_path / "config.json").write_text(json.dumps(dict(
        architecture="feature_cnn", image_size=16, dropout=0.3, grayscale=True)))
    sample = tmp_path / "sample.png"
    Image.fromarray((images[0, 0].numpy() * 255).astype("uint8")).save(sample)
    with Image.open(classifier.export_features(sample, tmp_path)) as image:
        assert image.width > 1000


def test_features_missing_image_reports_path_before_model_loading(tmp_path, monkeypatch):
    import pytest
    import image_classifier_bnn as classifier

    def unexpected_load(*args):
        raise AssertionError("Missing image should be checked before model loading")

    monkeypatch.setattr(classifier, "_load", unexpected_load)
    with pytest.raises(FileNotFoundError, match="missing.jpg"):
        classifier.export_features(tmp_path / "missing.jpg", tmp_path)
