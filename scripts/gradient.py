import tensorflow as tf

x = tf.Variable(3.5, dtype=tf.float32)
with tf.GradientTape(persistent=True) as tape:
    y = (x - 1) * (x - 2) * (x - 3)
    z = y ** 2
dy_dx = tape.gradient(y, x)
dz_dx = tape.gradient(z, x)
print(dy_dx)
print(dz_dx)
