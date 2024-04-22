import tensorflow as tf
from tensorflow.keras import layers

def normalize_l(in_l, l_cent=50., l_norm=100.):
    return (in_l - l_cent) / l_norm

def unnormalize_l(in_l, l_cent=50., l_norm=100.):
    return in_l * l_norm + l_cent

def normalize_ab(in_ab, ab_norm=110.):
    return in_ab / ab_norm

def unnormalize_ab(in_ab, ab_norm=110.):
    return in_ab * ab_norm

def ECCVGenerator(norm_layer=tf.keras.layers.BatchNormalization):
    input_l = tf.keras.Input(shape=(None, None, 1))
    conv1_2 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', bias=True)(normalize_l(input_l))
    conv1_2 = layers.ReLU()(conv1_2)
    conv1_2 = layers.Conv2D(64, kernel_size=3, strides=2, padding='same', bias=True)(conv1_2)
    conv1_2 = layers.ReLU()(conv1_2)
    conv1_2 = norm_layer()(conv1_2)

    conv2_2 = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', bias=True)(conv1_2)
    conv2_2 = layers.ReLU()(conv2_2)
    conv2_2 = layers.Conv2D(128, kernel_size=3, strides=2, padding='same', bias=True)(conv2_2)
    conv2_2 = layers.ReLU()(conv2_2)
    conv2_2 = norm_layer()(conv2_2)

    conv3_3 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', bias=True)(conv2_2)
    conv3_3 = layers.ReLU()(conv3_3)
    conv3_3 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', bias=True)(conv3_3)
    conv3_3 = layers.ReLU()(conv3_3)
    conv3_3 = layers.Conv2D(256, kernel_size=3, strides=2, padding='same', bias=True)(conv3_3)
    conv3_3 = layers.ReLU()(conv3_3)
    conv3_3 = norm_layer()(conv3_3)

    conv4_3 = layers.Conv2D(512, kernel_size=3, strides=1, padding='same', bias=True)(conv3_3)
    conv4_3 = layers.ReLU()(conv4_3)
    conv4_3 = layers.Conv2D(512, kernel_size=3, strides=1, padding='same', bias=True)(conv4_3)
    conv4_3 = layers.ReLU()(conv4_3)
    conv4_3 = layers.Conv2D(512, kernel_size=3, strides=1, padding='same', bias=True)(conv4_3)
    conv4_3 = layers.ReLU()(conv4_3)
    conv4_3 = norm_layer()(conv4_3)

    conv5_3 = layers.Conv2D(512, kernel_size=3, dilation_rate=2, strides=1, padding='same', bias=True)(conv4_3)
    conv5_3 = layers.ReLU()(conv5_3)
    conv5_3 = layers.Conv2D(512, kernel_size=3, dilation_rate=2, strides=1, padding='same', bias=True)(conv5_3)
    conv5_3 = layers.ReLU()(conv5_3)
    conv5_3 = layers.Conv2D(512, kernel_size=3, dilation_rate=2, strides=1, padding='same', bias=True)(conv5_3)
    conv5_3 = layers.ReLU()(conv5_3)
    conv5_3 = norm_layer()(conv5_3)

    conv6_3 = layers.Conv2D(512, kernel_size=3, dilation_rate=2, strides=1, padding='same', bias=True)(conv5_3)
    conv6_3 = layers.ReLU()(conv6_3)
    conv6_3 = layers.Conv2D(512, kernel_size=3, dilation_rate=2, strides=1, padding='same', bias=True)(conv6_3)
    conv6_3 = layers.ReLU()(conv6_3)
    conv6_3 = layers.Conv2D(512, kernel_size=3, dilation_rate=2, strides=1, padding='same', bias=True)(conv6_3)
    conv6_3 = layers.ReLU()(conv6_3)
    conv6_3 = norm_layer()(conv6_3)

    conv7_3 = layers.Conv2D(512, kernel_size=3, strides=1, padding='same', bias=True)(conv6_3)
    conv7_3 = layers.ReLU()(conv7_3)
    conv7_3 = layers.Conv2D(512, kernel_size=3, strides=1, padding='same', bias=True)(conv7_3)
    conv7_3 = layers.ReLU()(conv7_3)
    conv7_3 = layers.Conv2D(512, kernel_size=3, strides=1, padding='same', bias=True)(conv7_3)
    conv7_3 = layers.ReLU()(conv7_3)
    conv7_3 = norm_layer()(conv7_3)

    conv8_3 = layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same', bias=True)(conv7_3)
    conv8_3 = layers.ReLU()(conv8_3)
    conv8_3 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', bias=True)(conv8_3)
    conv8_3 = layers.ReLU()(conv8_3)
    conv8_3 = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', bias=True)(conv8_3)
    conv8_3 = layers.ReLU()(conv8_3)
    conv8_3 = layers.Conv2D(313, kernel_size=1, strides=1, padding='valid', bias=True)(conv8_3)

    softmax = layers.Softmax(axis=3)
    out_reg = softmax(conv8_3)
    out_reg = layers.Conv2D(2, kernel_size=1, padding='valid', dilation_rate=1, strides=1, use_bias=False)(out_reg)
    out_reg = unnormalize_ab(layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(out_reg))

    model = tf.keras.Model(inputs=input_l, outputs=out_reg)
    return model


