import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
np.set_printoptions(threshold=np.inf)

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255, x_test / 255

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3])

image_gen_train = ImageDataGenerator(
    rescale=1/1,
    rotation_range=45,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True,
    zoom_range=0.5
)
image_gen_train.fit(x_train)

class AlexNet(Model):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.c1 = Conv2D(filters=96, kernel_size=(3,3))
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.p1 = MaxPool2D(pool_size=(3,3), strides=2)

        self.c2 = Conv2D(filters=256, kernel_size=(3,3))
        self.b2 = BatchNormalization()
        self.a2 = Activation('relu')
        self.p2 = MaxPool2D(pool_size=(3,3), strides=2)

        self.c3 = Conv2D(filters=384, kernel_size=(3,3), padding='same', activation='relu')

        self.c4 = Conv2D(filters=384, kernel_size=(3,3), padding='same', activation='relu')

        self.c5 = Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu')
        self.p3 = MaxPool2D(pool_size=(3,3), strides=2)

        self.flatten = Flatten()
        self.f1 = Dense(2048, activation='relu')
        self.d1 = Dropout(0.5)
        self.f2 = Dense(2048, activation='relu')
        self.d2 = Dropout(0.5)
        self.f3 = Dense(10, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        inputs = self.c1(inputs)
        inputs = self.b1(inputs)
        inputs = self.a1(inputs)
        inputs = self.p1(inputs)

        inputs = self.c2(inputs)
        inputs = self.b2(inputs)
        inputs = self.a2(inputs)
        inputs = self.p2(inputs)

        inputs = self.c3(inputs)
        inputs = self.c4(inputs)
        inputs = self.c5(inputs)
        inputs = self.p3(inputs)

        inputs = self.flatten(inputs)
        inputs = self.f1(inputs)
        inputs = self.d1(inputs)
        inputs = self.f2(inputs)
        inputs = self.d2(inputs)
        y = self.f3(inputs)
        return y

model = AlexNet()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = './checkpoint/LeNet5.ckpt'
if os.path.exists(checkpoint_save_path + 'index'):
    print('------------------load the data---------------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)
history = model.fit(image_gen_train.flow(x_train, y_train, batch_size=32),
                    epochs=100, validation_data=(x_test, y_test),
                    validation_freq=1, callbacks=[cp_callback])
model.summary()

print(model.trainable_variables)
file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()