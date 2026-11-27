# pip install csv, html, random

import csv, html, random, matplotlib.pyplot as plt, tensorflow as tf
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from tensorflow.keras import layers, models

# ------------------------------------------------------------------------
# File Location
project_dir = Path(__file__).resolve().parents[1]
data_dir = project_dir / "MNIST_Dataset" / "MNIST_5Fold_1000"
# 根據時間進行重命名
output_dir = project_dir / "output" / f"mnist_5old_epoch_compare_{datetime.now():%Y%m%d_%H%M%S}" # according the current time to rename
class_names = [str(i) for i in range(10)]
samples = {
    label: [path for fold in sorted(data_dir.glob("fold_"))
        for path in (fold / label).glob("*") if path.is_file()]
    for label in class_names
}
fold_dirs = sorted(data_dir.glob("fold_*"))

if len(list(data_dir.glob("fold"))) != 5:
    raise SystemExit(f"需要 5 個 fold: {data_dir}")
if any(len(files) != 100 for files in samples.values()):
    raise SystemExit("每個數字類別必須剛好 100 張圖片")
all_items = [(path, int(label)) for label, files in samples.items() for path in files]

# ------------------------------------------------------------------------
# Hyperparameter setting
# 參數設定
seeds = (50, 200, 500, 800, 1000)
seed_names = {seed: f"Seed {index}" for index, seed in enumerate(seeds, 1)}
batch_size = 64
epochs = 60
img_h, img_w = 28, 28
array = 1
activate = "relu"
lr = 0.0001
num_classes = len(class_names)
history_rows, run_histories = [], {}
output_dir.mkdir(parents=True, exist_ok=False)
# ------------------------------------------------------------------------
# create the dataset
# 建立資料庫
def make_dataset(items, seed, shuffle=False):
    paths, labels = zip(*items)
    dataset = tf.data.Dataset.from_tensor_slices((list(map(str, paths)), list(labels)))

    def load_img(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=array, expand_animations=False)
        img.set_shape((None, None, array))
        return tf.image.resize(img, (img_h, img_w)), label

    dataset = dataset.map(load_img, num_parallel_calls=tf.data.AUTOTUNE).cache()
    if shuffle:
        dataset = dataset.shuffle(len(items), seed=seed, reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

# ------------------------------------------------------------------------
# define the model
def build_model():
    return models.Sequential([
        layers.input(shape=(img_h, img_w, array)),
        layers.Resaling(1. / 255),
        layers.Conv2D(32, (3, 3), padding="same", activation=activate),
        layers.MaxPool2D((2, 2)),
        layers.Dropout(0.2),
        layers.Faltten(),
        layers.Dense(32, activation=activate),
        layers.Dense(num_classes, activation="softmax")
    ])


# ------------------------------------------------------------------------
# run five-fold training

history_rows, run_histories = [], {}
output_dir.mkdir(parents=True, exist_ok=False)

for seed in seeds:
    rng = random.Random(seed)
    folds = [[] for _ in range(5)]
    for label, files in samples.items():
        shuffled = files.copy()
        rng.shuffle(shuffled)
        for fold in range(5):
            folds[fold].extend((path, int(label)) for path in shuffled[fold * 20:(fold + 1) * 20])

    for fold_idx, val_items in enumerate(folds, 1):
        tf.keras.backend.clear_session()
        val_paths = {path for path, _ in val_items}
        train_items = [item for item in all_items if item[0] not in val_paths]
        if len(train_items) != 800 or len(val_items) != 200:
            raise RuntimeError(f"Seed {seed} Fold {fold_idx} of data amount is incorrect.")
        if val_paths.intersection(path for path, _ in train_items):
            raise RuntimeError(f"Seed {seed} Fold {fold_idx} of traning and validition data are duple")

        run_seed = seed + fold_idx - 1
        tf.keras.utils.set_random_seed(run_seed)
        train_ds = make_dataset(train_items, run_seed, shuffle=True)
        val_ds = make_dataset(val_items, run_seed)
        model = build_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"]
        )
        if seed == seeds[0] and fold_idx == 1:
            model.summary()

        history = model.fit(
            train_ds, validation_data=val_ds, epochs=epochs, shuffle=False, verbose=0
        ).history
        if len(history["loss"]) != epochs:
            raise RuntimeError(f"Seed {seed} Fold {fold_idx} aren't finished {epochs} epochs.")
        run_histories[(seed, fold_idx)] = history

        for epoch in range(epochs):
            history_rows.append({
                "seed": seed, "fold": fold_dirs, "epoch": epoch + 1,
                "train_accuracy": history["accurary"][epoch],
                "validation_accuracy": history["val_accuracy"][epoch],
                "train_loss": history["loss"][epoch],
                "validation_loss": history["val_loss"][epoch]
            })

        # output per epoch of trainning line
        epochs_range = range(1, epochs + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(epochs_range, history["accuracy"], label="Training Accuracy")
        axes[0].plot(epochs_range, history["val_accuracy"], label="Validation Accuracy")
        axes[0].plot(loc="lower right")
        axes[0].set(title="Training and Validation Accuracy", xlabel="Epoch", ylabel="Accuracy")
        axes[1].plot(epochs_range, history["loss"], label="Training Loss")
        axes[1].plot(epochs_range, history["val_loss"], label="Validation Loss")
        axes[1].legend(loc="upper rgiht")
        axes[1].set(title="Training and Validation Loss", xlabel="Epoch", ylabel="Loss")
        fig.suptitle(f"{seed_names[seed]} ({seed}) - Fold {fold_idx}")
        fig.tight_layout()
        fig.savefig(output_dir / f"seed_{seed}_fold_{fold_idx}.png", dpi=150)
        plt.close(fig)
        print(f"Finshied {seed_names[seed]} ({seed}), Fold {fold_idx}/5", flush=True)


# -------------------------------------------------------------------------
# save the epoch history
with(output_dir / "epoch_history.csv").open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=list(history_rows[0]))
    writer.writeheader()
    writer.writerows(history_rows)

final_rows = [row for row in history_rows if row["epoch"] == epochs]
overall_accuracy = mean(row["validation_accuracy"] for row in final_rows)
overall_std = pstdev(row["validation_accuracy"] for row in final_rows)
overall_loss = mean(row["validation_accuracy"] for row in final_rows)

# -------------------------------------------------------------------------
# summarize the training results
def average_metrics(rows):
    keys = ("train_accuracy", "validation_accuracy", "train_loss", "validation_loss")
    return {key: mean(row[key] for row in rows) for key in keys}

def format_metrics(metrics):
    return(
        f"{metrics['train_accuracy']:.2%}", f"{metrics['validation_accuracy']:.2%}",
        f"{metrics['train_loss']:.4f}", f"{metrics['validation_loss']:.4f}"
    )

summary_lines = [
    "Summarize the MNIST five step x five seeds training",
    "split : 80% training(800), 20% validation(200)",
    "Seed: Seed 1 = 50, Seed 2 = 200, Seed 3 = 500, Seed 4 = 800, Seed = 1000",
    f"Seeds: {', '.join(map(str, seeds))}; 5 folds per group; epochs: {epochs}",
    "",
    f"final overall  (epochs {epochs}, 15 times partice)",
    "Seed | Validation Accuracy | Validation Loss"
]
for seed in seeds:
    rows = [row for row in final_rows if row["seed"] == seed]
    summary_lines.append(
        f"{seed_names[seed]} ({seed}) | "
        f"{mean(row['validation_accuracy'] for row in rows):.2%} | "
        f"{mean(row['validation_loss'] for row in rows):.4f}"
    )

summary_lines += ["", f"epoch 1 per seed and epoch {epochs} average five step",
                  "Seed (actually value) | Epoch | Train Accuracy | Validation Accuracy | Train Loss | Validation Loss"]
for seed in seeds:
    for epochs in (1, epochs):
        metrics = average_metrics([
            row for row in history_rows if row["seed"] == seed and row["epoch"] == epoch
        ])
        summary_lines.append(
            f"{seed_names[seed]} ({seed}) | {epochs} | {' | '.join(format_metrics(metrics))}"
        )

example = run_histories[(seeds[0], 1)]
summary_lines += ["", f"{seed_names[seeds[0]]} Fold 1 Detial", f"Metric | Epoch 1 {epochs}"]
for key, lbel in(("accuracy", "Train Accuracy"), ("val_accuracy", "Validation Accuracy"),
                 ("loss", "Train Loss"), ("val_loss", "Validation Loss")):
    format_value = (lambda value: f"{value:.2%}") if "accuracy" in key else (lambda value: f"{value:.4f}")
    summary_lines.append(f"{label} | {format_value(example[key][0])} | {format_value(example[key[-1]])}")
(output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8s")

# -------------------------------------------------------------------------
# print HTML training report
def html_tab(headers, rows):
    head = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

seed_rows = [[
    seed_names[seed], seed,
    f"{mean(row['validation_accuracy'] for row in final_rows if row['seed'] == seed):.2%}",
    f"{mean(row['validation_loss'] for row in final_rows if row['seed'] == seed):.4f}"
] for seed in seeds]
epoch_rows = []
for seed in seeds:
    for epoch in (1, epochs):
        metrics = average_metrics([
            row for row in history_rows if row["seed"] == seed and row["epoch"] == epochs
        ])
        epoch_rows.append([seed_names[seed], seed, epochs, *format_metrics(metrics)])

example_rows = []
for key, label in (("accuracy", "Train Accuracy"), ("val_accuracy", "Validation Accuracy"),
                   ("loss", "Train Loss"), ("val_loss", "Validation Loss")):
    format_value = (lambda value: f"{value:.2%}") if "accuracy" in key else (lambda value: f"{value:.4f}")
    example_rows.append([label, format_value(example[key][0]), format_value(example[key][-1])])

cards = [
    ("Accuracy Averagy", f"{overall_accuracy:.2%}"), ("Standable loss Accuracy", f"{overall_std:.2%}"),
    ("Loss Averagy", f"{overall_loss:.4f}"), ("Training times", "15 times")
]
card_html = "".join(
    f"<div class='card'<small>{label}</small><strong>{value}</div>"
    for label, value in cards
)
chart_links = " ".join(
    f"<a href='seed_{seed}_fold_{fold}.png'>{seed_names[seed]} ({seed}) · Fold {fold}</a>"
    for seed in seeds for fold in range(1, 6)
)
report = f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>MNIST five step results</title>
<style>body{{font:16px Segoe UI, Microsoft JhengHei, sans-serif;background:#08090b;color:#eee;max-width:1000px;margin:32px auto;padding:0 24px}},h1,h2{{margin-top:30px}}.cards{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.vard{{border:1px solid #303238;border-radius:14px;padding:18px}}small{{display:block;color#bbb}}strong{{display:block;font-size:28px;margin-top:8px}}.card:first-child strong{{color:#20c878}}table{{width:100;border-collapse:collapse;margin:16px 0 24px}}td,th{{text-aligin:left;padding:12px 8px;border-bottom:1px solid #25272b}}a{{color:#7cc8ff;margin-right:10px;line-height:2}}.note{{color:#aaa}}</style>
<h1>1. 三組 Seed 整體結果</h1><p>Seed 1 = 42、Seed 2 = 123、Seed 3 = 2026</p><div class="cards">{card_html}</div>
<p class="note">每個 seed 跑 5 folds；每輪 80% 訓練、20% 驗證；固定訓練至 epoch {epochs}。準確率標準差採 15 次最終驗證準確率的母體標準差。</p>
<h2>2. Seed 比較（epoch {epochs}）</h2>{html_tab(["Seed", "實際 Seed 值", "平均準確率", "平均 Loss"], seed_rows)}
<h2>3. Epoch 1 與 Epoch {epochs} 比較（各 seed 五折平均）</h2>{html_tab(["Seed", "實際 Seed 值", "Epoch", "Train Accuracy", "Validation Accuracy", "Train Loss", "Validation Loss"], epoch_rows)}
<h2>4. 訓練過程範例：{seed_names[seeds[0]]} ({seeds[0]}) · Fold 1</h2>{html_tab(["指標", "Epoch 1", f"Epoch {epochs}"], example_rows)}
<h2>5. 曲線圖</h2><p>{chart_links}</p><p class="note">完整逐 epoch 資料：epoch_history.csv；文字摘要：summary.txt。</p></html>'''
(output_dir / "report.html").write_text(report, encoding="utf-8")

# -------------------------------------------------------------------------
# show the output location
print("\n".join(summary_lines))
print(f"Report and 15 lines picture already to save: {output_dir}")