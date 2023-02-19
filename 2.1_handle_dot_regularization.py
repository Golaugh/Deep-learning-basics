import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('dataset/dot.csv')
x_data = np.array(df[['x1', 'x2']])
y_data = np.array(df['y_c'])
'''
print(x_data)
print(y_data)
print('---------------------------------------------------')
'''
x_train = np.vstack(x_data).reshape(-1, 2)
y_train = np.vstack(y_data).reshape(-1, 1)
'''
print(x_train)
print(y_train)
print('---------------------------------------------------')
'''

Y_c = [['red' if y else 'blue'] for y in y_train]

x_train = tf.cast(x_train, tf.float32)
y_train = tf.cast(y_train, tf.float32)
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

w1 = tf.Variable(tf.random.normal([2, 11]), dtype=tf.float32)
b1 = tf.Variable(tf.constant(0.01, shape=[11]))

w2 = tf.Variable(tf.random.normal([11, 1]), dtype=tf.float32)
b2 = tf.Variable(tf.constant(0.01, shape=[1]))

lr = 0.005
epoch = 1001
REGULARIZER = 0.03

'''
SGD, SGDM(include the momentum compared the SGD), 
Adagrad(introduce the two-level momentum compared the SGD), 
RMSProp(add the two-level momentum compared the SGD),
ADAM(introduce both the momentum of SGDM and the two-level momentum of RMSProp)
'''
for epoch in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            h1 = tf.matmul(x_train, w1) + b1
            h1 = tf.nn.relu(h1)
            y = tf.matmul(h1, w2) + b2

            loss_mse = tf.reduce_mean(tf.square(y_train - y))
            loss_regularization = []
            loss_regularization.append(tf.nn.l2_loss(w1))
            loss_regularization.append(tf.nn.l2_loss(w2))
            loss_regularization = tf.reduce_sum(loss_regularization)
            loss = loss_mse + REGULARIZER * loss_regularization

        variables = [w1, b1, w2, b2]
        grads = tape.gradient(loss, variables)
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])

    if epoch % 100 == 0:
        print(f"epoch: {epoch}, loss: {float(loss)}")

print('********** predict **********')
xx, yy = np.mgrid[-3:3:.1, -3:3:.1]
grid = np.c_[xx.ravel(), yy.ravel()]
grid = tf.cast(grid, tf.float32)

proba = []
for x_test in grid:
    h1 = tf.matmul([x_test], w1) + b1
    h1 = tf.nn.relu(h1)
    y = tf.matmul(h1, w2) + b2
    proba.append(y)

x1 = x_data[:, 0]
x2 = x_data[:, 1]

proba = np.array(proba).reshape(xx.shape)
plt.scatter(x1, x2, color=np.squeeze(Y_c))
plt.contour(xx, yy, proba, levels=[.5])
plt.show()