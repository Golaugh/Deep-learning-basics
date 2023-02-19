import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import os

input_word = [['a'], ['b'], ['c'], ['d'], ['e']]
w_to_id = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4}
transformer = OneHotEncoder(categories='auto', handle_unknown='ignore')
transformer.fit(input_word)
x_train = transformer.transform(input_word).toarray()
y_train = [1, 2, 3, 4, 0]
input_word = "abcde"

np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)

x_train = np.reshape(x_train, (len(x_train), 1, 5))
y_train = np.array(y_train)

model = tf.keras.Sequential([
    SimpleRNN(3),
    Dense(5, activation='softmax')
])
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/RNN_one_hot.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='loss')

history = model.fit(x_train, y_train, batch_size=32,
                    epochs=100, callbacks=[cp_callback])
model.summary()

# print(model.trainable_variables)
file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

###############################################    show   ###############################################

# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
loss = history.history['loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

preNum = int(input("Input the number of test alphabet : "))
for i in range(preNum):
    alpha = input("input test alphabet : ")
    alpha_bet_1 = transformer.transform([[alpha]]).toarray()
    alpha_bet_1 = np.reshape(alpha_bet_1, (1, 1, 5))
    result = model.predict([alpha_bet_1])
    pred = tf.argmax(result, axis=1)
    pred = int(pred)
    tf.print(alpha + '->' + input_word[pred])