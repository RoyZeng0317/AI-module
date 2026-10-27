"""獨立 MNIST 辨識：首次自行訓練並保存，之後載入模型，只輸出數字。"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import argparse, contextlib, io, numpy as np
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

tf.get_logger().setLevel('ERROR')
# ------------------------------------------------------------------
# read the dataset
project_dir = Path(__file__).resolve().parents[1]
test_img_dir = project_dir / 'mnist_test/p5.png'  # 改成你的圖片檔案路徑
train_data_dir = project_dir / 'MNIST_Dataset/MNIST_Training'
model_path = project_dir / 'tranning/mnist_runs/digit.keras'
# Hyperparameter setting
batch_size = 256
img_h = 28
img_w = 28
array = 1
activate = 'relu'
lr = 0.0001
num_classes = 10
epochs = 50


# ------------------------------------------------------------------
# prepare the test image
def prepare_image(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert('RGBA')
        background = Image.new('RGBA', image.size, 'white')
        image = np.asarray(Image.alpha_composite(background, image).convert('L'))
    border = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]))
    if np.median(border) > 127:
        image = 255 - image
    y, x = np.where(image > 40)
    if not len(x) or np.ptp(image) < 10:
        raise ValueError('圖片沒有清楚的數字，請使用單一數字的圖片。')
    digit = Image.fromarray(image[y.min():y.max() + 1, x.min():x.max() + 1])
    # 小畫家極細筆畫直接縮圖會消失；只有稀疏筆畫才先加粗。
    if np.mean(np.asarray(digit) > 40) < 0.08:
        stroke = max(1, round(max(digit.size) / 10)) | 1
        digit = ImageOps.expand(digit, border=stroke // 2).filter(ImageFilter.MaxFilter(stroke))
    digit.thumbnail((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new('L', (img_w, img_h))
    canvas.paste(digit, ((img_w - digit.width) // 2, (img_h - digit.height) // 2))
    # 模型內已有 Rescaling，因此這裡保留 0～255。
    return np.asarray(canvas, dtype=np.float32).reshape(1, img_h, img_w, array)


# ------------------------------------------------------------------
# load the saved model or train a new model
def get_model(retrain=False):
    if model_path.is_file() and not retrain:
        return tf.keras.models.load_model(model_path, compile=False)
    # read and split the dataset
    with contextlib.redirect_stdout(io.StringIO()):
        train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
            train_data_dir, class_names=[str(i) for i in range(num_classes)],
            validation_split=0.2, subset='both', seed=123,
            image_size=(img_h, img_w), color_mode='grayscale', batch_size=batch_size,
        )
    # ------------------------------------------------------------------
    # define the model
    model = Sequential([
        layers.Input(shape=(img_h, img_w, array)),
        layers.Rescaling(1 / 255),
        layers.Conv2D(32, 3, padding='same', activation=activate),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(32, activation=activate),
        layers.Dense(num_classes, activation='softmax'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # ------------------------------------------------------------------
    # train and save the model
    model.fit(train_ds.prefetch(1), validation_data=val_ds.prefetch(1),
              epochs=epochs, verbose=0, shuffle=False)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    return model


# ------------------------------------------------------------------
# predict the test image
def main(retrain=False):
    image = prepare_image(test_img_dir)
    scores = get_model(retrain).predict(image, verbose=0)[0]
    print(int(np.argmax(scores)))


# ------------------------------------------------------------------
# command line options
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='辨識單張手寫數字，只輸出數字。')
    parser.add_argument('--retrain', action='store_true', help='依目前 epochs 重新訓練並更新模型')
    main(parser.parse_args().retrain)
