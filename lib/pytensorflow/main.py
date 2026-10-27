# pip install numpy, tensorflow
# .\conda\python.exe -m pip install numpy, tensorflow
# 前置作業: 準備 MNIST 資料
import numpy as np, tensorflow as tf

def dense(input_layer, units, activation="relu"):
    return tf.keras.layers.Dense(units, activation=activation)(input_layer)
def dropout(input_layer, rate, training=None):
    return tf.keras.layers.Dropout(rate)(input_layer, training=training)
def conv2d(input_layer, filters):
    return tf.keras.layers.Conv2D(filters, kernel_size=(5, 5), strides=1,
                                  padding="same", activation="relu")(input_layer)
def pooling2d(input_layer):
    return tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(input_layer)

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # 舊教學會將 60,000 張分成 55,000 訓練與 5,000 驗證。
    x_val, y_val = x_train[:5000], y_train[:5000]
    x_train, y_train = x_train[5000:], y_train[5000:]
    x_test, y_test = x_test[:4000], y_test[:4000]
    # 與舊教學一致：影像攤平成 784 個特徵、縮放至 0~1，標籤為 one-hot。
    x_train, x_val, x_test = [x.reshape(-1, 784).astype(np.float32) / 255
                             for x in (x_train, x_val, x_test)]
    y_train, y_val, y_test = [tf.keras.utils.to_categorical(y, num_classes=10)
                             for y in (y_train, y_val, y_test)]
# 設定神經網路模型輸入、輸出與測試資料

# 資料前處理已除以 255；Keras 訓練時直接傳入 y_train，不需要標籤 placeholder。
x_input = tf.keras.Input(shape=(784,), dtype="float32", name="x_input")
x_input_images = tf.keras.layers.Reshape((28, 28, 1))(x_input)

conv1 = conv2d(input_layer=x_input_images, filters=32)
pool1 = pooling2d(input_layer=conv1)
conv2 = conv2d(input_layer=pool1, filters=64)
pool2 = pooling2d(input_layer=conv2)

flat_0 = tf.keras.layers.Flatten()(pool2)
dense_0 = dense(input_layer=flat_0, units=1024)
dropout_0 = dropout(input_layer=dense_0, rate=0.4)
dense_1 = dense(input_layer=dropout_0, units=1024)
dropout_1 = dropout(input_layer=dense_1, rate=0.6)
# logits 是未經 softmax 的分數，輸出層不使用 ReLU。
logits = dense(input_layer=dropout_1, units=10, activation=None)

# 設定損失函數優化器與學習率
model = tf.keras.Model(inputs=x_input, outputs=logits)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.000095)
accuracy = tf.keras.metrics.CategoricalAccuracy(name="accuracy")
model.compile(optimizer=optimizer, loss=loss, metrics=[accuracy])
# 訓練時由 model.fit(x_train, y_train, validation_data=(x_val, y_val)) 傳入標籤。
