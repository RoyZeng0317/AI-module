# pip install matplotlib, numpy, tensorflow, opencv-python
# .\.conda\python.exe -m pip install matplotlib
# .\.conda\python.exe -m pip install tensorflow
import cv2, numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential

# ------------------------------------------------------------------
# read the dataset
test_img_dir = Path(__file__).resolve().parents[1] / "mnist_test/p1.png"   # test image file：改成小畫家圖片的完整路徑
project_dir = Path(__file__).resolve().parents[1]
train_data_dir = project_dir / "MNIST_Dataset/MNIST_Training"    # also can enter absoulte location
validation_data_dir = train_data_dir    # from same folder to split the 20% to verify set

# Hyperparameter setting
batch_size = 256
img_h = 28
img_w = 28
array = 1
activate = 'relu'
lr = 0.001
num_classes = 10
epochs = 30

# ------------------------------------------------------------------
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    train_data_dir,
    validation_split=0.2,
    subset="both",
    seed=123,
    image_size=(img_h, img_w),
    batch_size=batch_size,
    color_mode="grayscale"
)
# ------------------------------------------------------------------
# make sure the types
class_names = train_ds.class_names
num_classes = len(class_names)
print(class_names)  # ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'] total has 10 types

# ------------------------------------------------------------------
# make sure the size

for img_batch, labels_batch in train_ds:
    print(img_batch.shape)
    print(labels_batch.shape)
    break

# ------------------------------------------------------------------
# dataset to abaility

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ------------------------------------------------------------------
# define the model
model = Sequential([
    # layers.Recaleing()
    layers.Input(shape=(img_h, img_w, array)),
    layers.Rescaling(1/255),
    # convosing
    layers.Conv2D(32, (3, 3), padding='same', activation=activate),
    # pooling layer (pool_size)
    layers.MaxPool2D((2, 2)),
    # Dropout = close the random dropout(amount)
    layers.Dropout(0.2),
    # 
    layers.Flatten(),
    # Full Connection Layer
    layers.Dense(32, activation=activate),
    layers.Dense(num_classes, activation='softmax')
])

# ------------------------------------------------------------------
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),     # setting the optimizer and Learning Rate
              loss=tf.keras.losses.SparseCategoricalCrossentropy(       # setting the loss
                    from_logits=False),
                metrics=['accuracy'])
model.summary()     # show the model summary

# ------------------------------------------------------------------
# starting to training the model
history = model.fit(                                                    # the model and dataset to match and setting the training epochs
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# ------------------------------------------------------------------
# print the image to check the training result easier
acc = history.history['accuracy']   # get the 
val_acc = history.history['val_accuracy']   # get the varify 

loss = history.history['loss']  # get the training of loss value
val_loss = history.history['val_loss']  # get the verifyset of loss value

epochs_range = range(epochs)    # setting the training time range

plt.figure(figsize=(8, 8))  # setting the image size
plt.subplot(1, 2, 1)    # setting the image location
plt.plot(epochs_range, acc, label='Training Accuracy')  # draw the traningset to
plt.plot(epochs_range, val_acc, label='Validation Accuracy')    # draw the verifyset to 
plt.legend(loc='lower right')   # set the image locate
plt.title('Training and Validation Accuracy')   # set the image title

plt.subplot(1, 2, 2)    # set the image locate
plt.plot(epochs_range, loss, label='Traning Loss')  # draw the trainingset loss 
plt.plot(epochs_range, val_loss, label='Validation Loss')   # draw the verifyset loss
plt.legend(loc='upper right')   # set locate
plt.title('Training and Validation Loss')   # set the image title
plt.show()  # show the image

# ------------------------------------------------------------------
# choose a test data to test
img_path = Path(test_img_dir)  # can choose the abosulte locate
if img_path.is_file():
    img = tf.keras.utils.load_img(
        img_path, target_size=(img_h, img_w),
        color_mode="grayscale"
    )   # read the image and convert to gray
    img_array = tf.keras.utils.img_to_array(img)    # convert to matrix
    # MNIST 是黑底白字；小畫家若使用白底黑字，依圖片邊框的底色反轉。
    border = np.concatenate((img_array[0].ravel(), img_array[-1].ravel(),
                             img_array[:, 0].ravel(), img_array[:, -1].ravel()))
    if np.median(border) > 127:
        img_array = 255 - img_array
    if np.ptp(img_array) < 10:
        raise ValueError(f"圖片沒有清楚的數字筆畫，請確認不是空白圖片：{img_path}")
    img_array = tf.expand_dims(img_array, 0)    # create a batch

    predictions = model.predict(img_array, verbose=0)  # predict the image
    score = predictions[0]  # model output already is a softmax 
    print(score)
    print(
        "辨識結果：{}，模型預測機率：{:.2f}%"
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
else:
    print(f"Can't find the test image, skip a single to predict: {img_path}")
