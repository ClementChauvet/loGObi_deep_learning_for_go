from tensorflow.keras import layers 
from tensorflow.keras import regularizers


def CA_Dense(inputs, name = "CA", ratio=8):
    inputs = layers.Reshape((1,inputs.shape[1],inputs.shape[2],inputs.shape[3]))(inputs)
    w, h, d, out_dim = [int(x) for x in inputs.shape[1:]]
    temp_dim = max(int(out_dim // ratio), ratio)
 
    h_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 3]))(inputs)
    w_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=[2, 3]))(inputs)
    d_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2]))(inputs)
 
    x = layers.Concatenate(axis=1)([w_pool, h_pool, d_pool])
    x = layers.Reshape((1, 1, w + h + d, out_dim))(x)
    x = layers.Conv3D(temp_dim, 1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x_w, x_h, x_d = layers.Lambda(lambda x: tf.split(x, [w, h, d], axis=3))(x)
    x_w = layers.Reshape((w, 1, 1, temp_dim))(x_w)
    x_d = layers.Reshape((1, 1, d, temp_dim))(x_d)
    x_h = layers.Reshape((1, h, 1, temp_dim))(x_h)
 
    x_w = layers.Conv3D(out_dim, 1, activation='sigmoid')(x_w)
    x_h = layers.Conv3D(out_dim, 1, activation='sigmoid')(x_h)
    x_d = layers.Conv3D(out_dim, 1, activation='sigmoid')(x_d)
    x = layers.Multiply()([inputs, x_w, x_h, x_d])
    x = layers.Reshape((inputs.shape[2],inputs.shape[3],inputs.shape[4]))(x)
    return x


def CA(inputs, name = "CA", ratio=8):
    print(inputs.shape)
    w, h,out_dim = [int(x) for x in inputs.shape[1:]]
    temp_dim = max(int(out_dim // ratio), ratio)
 
    h_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 3]))(inputs)
    w_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=[2, 3]))(inputs)
 
    x = layers.Concatenate(axis=1)([w_pool, h_pool])
    x = layers.Reshape((1, 1, w + h))(x)
    x = layers.Conv2D(temp_dim, 1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    print(x,w,h)
    x_w, x_h= layers.Lambda(lambda x: tf.split(x, [w, h], axis=2))(x)
    x_w = layers.Reshape((1,1,w*temp_dim))(x_w)
    x_h = layers.Reshape((1,1, h, temp_dim))(x_h)
 
    x_w = layers.Dense(out_dim, activation='sigmoid')(x_w)
    x_h = layers.Dense(out_dim, activation='sigmoid')(x_h)
    x = layers.Multiply()([inputs, x_w, x_h])
    return x
