import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255

class FashionMnist(Model):
    def __init__(self):
        super(FashionMnist, self).__init__()
        self.f1 = tf.keras.layers.Flatten()
        self.d1 = Dense(128, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2())
        self.d2 = Dense(64, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2())
        self.d3 = Dense(32, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2())
        self.d4 = Dense(10, activation='softmax',
                        kernel_regularizer=tf.keras.regularizers.l2())
    def call(self, inputs, training=None, mask=None):
        y = self.f1(inputs)
        y = self.d1(y)
        y = self.d2(y)
        y = self.d3(y)
        return self.d4(y)

model = FashionMnist()
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
model.fit(x_train, y_train, batch_size=32,
          epochs=50, validation_split=0.2,
          validation_freq=1)
model.summary()