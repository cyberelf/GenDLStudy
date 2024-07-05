"""卷积和反卷积"""

import numpy as np

from keras import layers


x = np.random.rand(4, 8, 8, 128)
conv = layers.Conv2D(32, 2, strides=2, padding='same', activation='relu')
convt = layers.Conv2DTranspose(32, 2, strides=2, padding='same', activation='relu')

y = conv(x)
z = convt(x)

print(conv.kernel.shape)
print(convt.kernel.shape)

print(y.shape)
print(z.shape)
