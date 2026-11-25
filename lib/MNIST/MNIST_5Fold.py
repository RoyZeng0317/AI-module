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
output_dir = project_dir / "output" / f"mnist_5old_epoch_compare_{datetime.now:%Y%m%d_%H%M%S}" # according the current time to rename
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
seeds = (42, 123, 2026)
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
        