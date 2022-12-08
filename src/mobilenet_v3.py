from tensorflow.keras import layers 
from tensorflow.keras import regularizers
import tensorflow as tf
from tensorflow import keras

def mixconv_block(t, filters):
    splitted = tf.split(t, num_or_size_splits=3, axis = 3)
    L = []
    for i in range(len(splitted)):
        L.append(layers.DepthwiseConv2D((i*2+1,i*2+1), padding="same",
            kernel_regularizer=regularizers.l2(0.00001),
            use_bias = False)(splitted[i]))
    t = tf.concat(L, axis = 3)
    return t
    
def SE_Block(t , filters , ratio =20): 
    se_shape = (1, 1, filters )
    squeeze = int(filters/ratio)
    se = layers.GlobalAveragePooling2D ()( t )
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense( squeeze,
        activation="swish",
        use_bias=True)(se)
    se = layers.Dense( filters,
        activation="sigmoid" ,
        use_bias=True)(se) 
    x = layers.multiply([t,se])
    return x


def bottleneck_block(x, expand, trunk, squeeze =False, kernel_DW = (3,3)):
    m = layers.Conv2D(expand, (1,1),
            kernel_regularizer=regularizers.l2(0.00001),
            use_bias = False)(x)
    m = layers.BatchNormalization()(m)
    m = layers.Activation("swish")(m)
    m = mixconv_block(m, expand)
    m = layers.BatchNormalization()(m)
    m = layers.Activation("swish")(m)
    if squeeze:
        m = SE_Block(m, expand)
    m = layers.Conv2D(trunk, (1,1),
            kernel_regularizer=regularizers.l2(0.00001),
            use_bias = False)(m)
    m = layers.BatchNormalization()(m)

    return layers.Add()([m, x])


def first_layer(inp, trunk):
    x = layers.Conv2D(trunk, (5,5), padding="same",
            kernel_regularizer=regularizers.l2(0.00001))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x1 = layers.Conv2D(trunk, (3,3), padding="same",
            kernel_regularizer=regularizers.l2(0.00001))(inp)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation("swish")(x1)
    x = layers.Add()([x,x1])
    return x
    
def siamese_head(x):
    policy_head = layers.Conv2D(1, 1, activation="swish", padding="same",
            use_bias = False,
            kernel_regularizer=regularizers.l2(0.00001))(x)
    policy_head = layers.Flatten()(policy_head)
    policy_head = layers.Activation("softmax", name="policy")(policy_head)
    value_head = layers.GlobalAveragePooling2D()(x)
    value_head = layers.Dense(50, activation="swish",
            kernel_regularizer=regularizers.l2(0.00001))(value_head)
    value_head = layers.Dense(1, activation="sigmoid", name="value",
    kernel_regularizer=regularizers.l2(0.00001))(value_head)
    
    return value_head, policy_head


if __name__ == '__main__':
    from training import train
    def getMobileNet_V3(blocks, filters = 294, trunk = 80):
        input = keras.Input(shape=(19, 19, 31), name="board")
        x = first_layer(input, trunk)
        for i in range (blocks):
            x = bottleneck_block (x, filters, trunk, squeeze = True)
        value_head, policy_head = siamese_head(x)
        model = keras.Model(inputs=input, outputs=[policy_head, value_head])
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.05),
              loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
              loss_weights={'policy' : 1.0, 'value' : 1.0},
              metrics={'policy': 'categorical_accuracy', 'value': 'mse'})
        return model
    model = getMobileNet_V3(15)
    model.summary()
    train(model, 8000, "MobileNetV3", 128)
    