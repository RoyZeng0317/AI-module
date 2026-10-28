# 輸入函示庫
import matplotlib.pyplot as plt
import numpy as np                                         
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from keras import layers, models
from keras.models import Sequential
from keras.applications.resnet import preprocess_input
import cv2

'''
環境套件安裝

1. pip install matplotlib
2. pip install tensorflow==2.6.0
3. pip install keras==2.6.0
4. pip install opencv-python
'''


#---------------------------------------------------------------------------------------------------------------------#
# 讀取資料集
train_data_dir = r'C:\Users\甲魚\Desktop\大三專題\123\MNIST_Dataset\MNIST_Training' # 也可以輸入絕對路徑
validation_data_dir = r'C:\Users\甲魚\Desktop\大三專題\123\MNIST_Dataset\MNIST_Validation'

# 參數設定
batch_size = 256  # 批量
img_height = 28  # 照片高度
img_width = 28  # 照片寬度
array = 1  # 維度
activate = 'relu'  # 激勵函數
lr = 0.0001  # Learning_rate 學習率
num_classes = 10  # 分類數量
epochs = 30  # 訓練次數

#---------------------------------------------------------------------------------------------------------------------#
# 訓練集與測試集設定
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="grayscale"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    validation_data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="grayscale"
)

#---------------------------------------------------------------------------------------------------------------------#
# 確認類別
class_names = train_ds.class_names
print(class_names)  # ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] 共十個類別

#---------------------------------------------------------------------------------------------------------------------#
# 確認尺寸
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

#---------------------------------------------------------------------------------------------------------------------#
# 配置數據集來調整性能
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#---------------------------------------------------------------------------------------------------------------------#
# 定義模型
model = Sequential([
    # layers.Rescaling( 正規化 , input_shape = (長 , 寬 , 維度)
    layers.Rescaling(1./255, input_shape=(img_height, img_width, array)),
    # 卷積層 (神經元數量 , 尺寸 , padding = 'valid' or 'same' , strides = 步數 , activate = 激勵函數 ('sigmoid' or 'tanh'))
    layers.Conv2D(32, 3, padding='valid', strides=(1, 1), activation=activate),
    # 池化層 (pool_size = 池化尺寸)
    layers.MaxPooling2D(pool_size=(2, 2)),
    # Dropout = 隨機關閉神經元 (數量)
    layers.Dropout(0.2),
    # 扁平層
    layers.Flatten(),
    # full connection layer 全連接層
    layers.Dense(32, activation=activate),
    layers.Dense(num_classes, activation='softmax')
])

#---------------------------------------------------------------------------------------------------------------------#
# 訓練模型配置
model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),               # 設定優化器以及 Learning_rate
              loss=keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),  # 設定 loss 計算方式
              metrics=['accuracy'])                                               # 設定 accuracy 計算方式
model.summary()

#---------------------------------------------------------------------------------------------------------------------#
# 進行模型訓練
history = model.fit(                                                             # 模型與資料集做結合 (match) 和設定訓練次數
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

#---------------------------------------------------------------------------------------------------------------------#
# 輸出圖形以便觀察訓練成果
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()

#---------------------------------------------------------------------------------------------------------------------#
# 選擇測試資料做測試
img_path = "./Test_data/5.jpg"  # 可以選擇絕對路徑
img = tf.keras.utils.load_img(
    img_path, target_size=(img_height, img_width),
    grayscale=True
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(score)
print(
    "這個可能是 {} 的機率是 {:.2f} %."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)


