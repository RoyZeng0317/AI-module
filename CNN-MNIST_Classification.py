# 匯入函式庫
# import the libraries
# pip install matplotlib numpy tensorflow opencv-python
# import matplotlib.pyplot as plt, numpy as np, tensorflow as tf, tensorflow.keras.backend as K
# .\.conda\python.exe -m pip install matplotlib
# .\.conda\python.exe -m pip install tensorflow
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
# import tensorflow.keras.backend as K (舊版示範)
# 在最新版的 python 中，keras 已被整合到 tensorflow 中，所以我們以直接使用 tensorflow.keras 來匯入 keras 模組
# 因為 Keras 已經是獨立到件，(安裝 tensorflow 時會一起裝好)
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
# from tensorflow.python.keras.applications.resent import preprocess_input  (舊版示範)
# 在新版的 keras 中，ResNet50 已經被整合到 keras.applications 中，所以我們以直接使用 keras.applications 來匯入 ResNet50 模組
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input, decode_predictions
from keras import backend as K


# ------------------------------------------------------------------------
# read the dataset
# 讀取資料集
project_dir = Path(__file__).resolve().parents[1]
train_data_dir = project_dir / "MNIST_Dataset/MNIST_Training"   # also can enter absolute location (也可以輸入絕對路徑)
validation_data_dir = train_data_dir  # 從同一資料夾切出 20% 作為驗證集

# Hyperparameter setting
# 參數設定
batch_size = 256    # batch size (批量)
img_height = 28     # image height  (影像高度)
img_width =28       # image width   (影像寬度)
array = 1           # array (陣列)
activate = 'relu'   # activa function, we use the ReLU (激活函數，我們使用 ReLU)
lr = 0.0001         # Learning_rate learnning (學習率)
num_classes = 10    # types amount (類別數量)
epochs = 30         # tranning times (訓練次數)

# ------------------------------------------------------------------------
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    train_data_dir,
    validation_split=0.2,
    subset="both",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="grayscale"
)

# ------------------------------------------------------------------------
# make sure the types
class_names = train_ds.class_names
num_classes = len(class_names)
print(class_names) # ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'] total has 10 types

# ------------------------------------------------------------------------
# make sure size

for img_batch, labels_batch in train_ds:
    print(img_batch.shape)
    print(labels_batch.shape)
    break

# ------------------------------------------------------------------------
#  dataset to  abaility
# 
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ------------------------------------------------------------------------
# define the model
model = Sequential([
    # layers.Rescaling(正規畫, input_shape=(長, 寬, 維度)),  # 影像歸一化
    layers.Input(shape=(img_height, img_width, array)),
    layers.Rescaling(1./255),
    # 捲積層 (神經元數量, 尺寸, padding = 'valid' or 'same', strides = 步數, activate = 激活函數 ('sigmoid' or 'tanh'))
    layers.Conv2D(32, (3, 3), padding='same', activation=activate),
    # 池化層 (pool_size = 池化尺寸)
    layers.MaxPooling2D((2, 2)),  # 池化層 (尺寸)
    # Dropout = 隨機關閉神經元 (數量)
    layers.Dropout(0.2),
    # 扁平層
    layers.Flatten(),
    # full connection layer 全連接層
    layers.Dense(32, activation=activate),
    layers.Dense(num_classes, activation='softmax')
])

# ------------------------------------------------------------------------
# 訓練模型配置
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),     # 設定優化器以及 Learning_rate
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=False),  # 設定 loss 計算方式
              metrics=['accuracy'])                                     # 設定 accuracy 計算方式
model.summary()  # 顯示模型摘要

# ------------------------------------------------------------------------
# 進行模型訓練
history = model.fit(                                                    # 模型與資料集做結合 (match) 和設定訓練次數
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# ------------------------------------------------------------------------
# 輸出圖形以便觀察訓練成果
acc = history.history['accuracy']  # 取得訓練集的準確率
val_acc = history.history['val_accuracy']  # 取得驗證集的準確率

loss = history.history['loss']  # 取得訓練集的損失值
val_loss = history.history['val_loss']  # 取得驗證集的損失值

epochs_range = range(epochs)  # 設定訓練次數範圍

plt.figure(figsize=(8, 8))  # 設定圖形大小
plt.subplot(1, 2, 1)  # 設定圖形位置
plt.plot(epochs_range, acc, label='Training Accuracy')  # 繪製訓練集準確率曲線
plt.plot(epochs_range, val_acc, label='Validation Accuracy')  # 繪製驗證集準確率曲線
plt.legend(loc='lower right')  # 設定圖例位置
plt.title('Training and Validation Accuracy')  # 設定圖形標題

plt.subplot(1, 2, 2)  # 設定圖形位置
plt.plot(epochs_range, loss, label='Training Loss')  # 繪製訓練集損失值曲線
plt.plot(epochs_range, val_loss, label='Validation Loss')  # 繪製驗證集損失值曲線
plt.legend(loc='upper right')  # 設定圖例位置
plt.title('Training and Validation Loss')  # 設定圖形標題
plt.show()  # 顯示圖形

# ------------------------------------------------------------------------
# 選擇測試資料做測試
img_path = project_dir / "Test_data/5.jpg"  # 可以選擇絕對路徑
if img_path.is_file():
    img = tf.keras.utils.load_img(
        img_path, target_size=(img_height, img_width),
        color_mode="grayscale"
    )  # 讀取影像並轉換為灰階
    img_array = tf.keras.utils.img_to_array(img)  # 轉換為矩陣
    img_array = tf.expand_dims(img_array, 0)  # create a batch (建立一個批次)

    predictions = model.predict(img_array)  # 預測影像
    score = predictions[0]  # 模型輸出已經是 softmax 機率
    print(score)
    print(
        "這個可能是 {} 的機率是 {:.2f} %"
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
else:
    print(f"找不到測試圖片，略過單張預測：{img_path}")
