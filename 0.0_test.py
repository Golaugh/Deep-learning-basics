import tensorflow as tf

tensorflow_version = tf.__version__
gpu_available = tf.test.is_gpu_available()

a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([1.0, 2.0], name="b")
result = tf.add(a, b, name="add")
print(f"tensorflow version: {tensorflow_version}")
print(f"GPU available: {gpu_available}")
print(result)