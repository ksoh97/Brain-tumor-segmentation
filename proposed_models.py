from tensorflow.keras.layers import *
import tensorflow as tf
import tensorflow.keras as keras
import proposed_layers as layers

class proposed_net:
    def __init__(self, ch, mode):
        self.mode, self.ch = mode, ch
        self.n_filters = [16, 32, 64, 128, 256]
        self.input = layers.input_layer(input_shape=(3, 256, 256), name="input")
        tf.keras.backend.set_image_data_format("channels_first")
        self.build_model()

    def conv_bn_act(self, x, f, n, s=1, k=None, p="same", batch=True, act=True, trans=False, out_p="auto"):
        if trans:
            c_layer = layers.conv_transpose
        else:
            c_layer = layers.conv

        if k:
            conv_l = c_layer(f=f, p=p, k=k, s=s, out_p=out_p, name=n + "_conv")
        else:
            conv_l = c_layer(f=f, p=p, name=n + "_conv")

        out = conv_l(x)
        if batch:
            norm_l = layers.batch_norm(name=n + "_norm")
            out = norm_l(out)

        if act:
            act_l = layers.relu(name=n + "_relu")
            out = act_l(out)
        return out

    def conv_bn_leakyact(self, x, f, n, s=1, k=None, p="same", act=True, trans=False, out_p="auto"):
        if trans:
            c_layer = layers.conv_transpose
        else:
            c_layer = layers.conv

        if k:
            conv_l = c_layer(f=f, p=p, k=k, s=s, out_p=out_p, name=n + "_conv")
        else:
            conv_l = c_layer(f=f, p=p, name=n + "_conv")
        out = conv_l(x)

        norm_l = layers.batch_norm(name=n + "_norm")
        out = norm_l(out)

        if act:
            act_l = layers.leaky_relu(name=n + "_leakyrelu")
            out = act_l(out)
        return out

    def concat(self, x, y, n):
        concat_l = layers.concat(name=n + "_concat")
        return concat_l([x, y])

    def batchnorm_relu(self, inputs):
        """ Batch Normalization & ReLU """
        x = BatchNormalization()(inputs)
        x = Activation("relu")(x)
        return x

    def residual_block(self, inputs, num_filters, strides=1):
        """ Convolutional Layers """
        x = self.batchnorm_relu(inputs)
        x = Conv2D(num_filters, 3, padding="same", strides=strides)(x)
        x = self.batchnorm_relu(x)
        x = Conv2D(num_filters, 3, padding="same", strides=1)(x)

        """ Shortcut Connection (Identity Mapping) """
        s = Conv2D(num_filters, 1, padding="same", strides=strides)(inputs)

        """ Addition """
        x = x + s
        return x

    def decoder_block(self, inputs, skip_features, num_filters):
        """ Decoder Block """

        x = UpSampling2D((2, 2))(inputs)
        x = Concatenate(axis=1)([x, skip_features])
        x = self.residual_block(x, num_filters, strides=1)
        return x

    def squeeze_excite_block(self, inputs, ratio=8):
        init = inputs
        channel_axis = 1
        filters = init.shape[channel_axis]
        se_shape = (filters, 1, 1)

        se = GlobalAveragePooling2D()(init)
        se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
        se = Reshape(se_shape)(se)

        x = Multiply()([init, se])
        return x

    def stem_block(self, x, n_filter, strides):
        x_init = x

        ## Conv 1
        x = Conv2D(n_filter, (3, 3), padding="same", strides=strides)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(n_filter, (3, 3), padding="same")(x)

        ## Shortcut
        s = Conv2D(n_filter, (1, 1), padding="same", strides=strides)(x_init)
        s = BatchNormalization()(s)

        ## Add
        x = Add()([x, s])
        x = self.squeeze_excite_block(x)
        return x

    def resnet_block(self, x, n_filter, strides=1):
        x_init = x

        ## Conv 1
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(n_filter, (3, 3), padding="same", strides=strides)(x)
        ## Conv 2
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(n_filter, (3, 3), padding="same", strides=1)(x)

        ## Shortcut
        s = Conv2D(n_filter, (1, 1), padding="same", strides=strides)(x_init)
        s = BatchNormalization()(s)

        ## Add
        x = Add()([x, s])
        x = self.squeeze_excite_block(x)
        return x

    def aspp_block(self, x, num_filters, rate_scale=1):
        x1 = Conv2D(num_filters, (3, 3), dilation_rate=(6 * rate_scale, 6 * rate_scale), padding="same")(x)
        x1 = BatchNormalization()(x1)

        x2 = Conv2D(num_filters, (3, 3), dilation_rate=(12 * rate_scale, 12 * rate_scale), padding="same")(x)
        x2 = BatchNormalization()(x2)

        x3 = Conv2D(num_filters, (3, 3), dilation_rate=(18 * rate_scale, 18 * rate_scale), padding="same")(x)
        x3 = BatchNormalization()(x3)

        x4 = Conv2D(num_filters, (3, 3), padding="same")(x)
        x4 = BatchNormalization()(x4)

        y = Add()([x1, x2, x3, x4])
        y = Conv2D(num_filters, (1, 1), padding="same")(y)
        return y

    def attetion_block(self, g, x):
        """
            g: Output of Parallel Encoder block
            x: Output of Previous Decoder block
        """

        filters = x.shape[1]

        g_conv = BatchNormalization()(g)
        g_conv = Activation("relu")(g_conv)
        g_conv = Conv2D(filters, (3, 3), padding="same")(g_conv)

        g_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(g_conv)

        x_conv = BatchNormalization()(x)
        x_conv = Activation("relu")(x_conv)
        x_conv = Conv2D(filters, (3, 3), padding="same")(x_conv)

        gc_sum = Add()([g_pool, x_conv])

        gc_conv = BatchNormalization()(gc_sum)
        gc_conv = Activation("relu")(gc_conv)
        gc_conv = Conv2D(filters, (3, 3), padding="same")(gc_conv)

        gc_mul = Multiply()([gc_conv, x])
        return gc_mul

    def build_model(self):
        if self.mode == 0:  # Unet
            enc_conv1_1 = self.conv_bn_act(self.input, f=self.ch, k=3, s=1, p="same", act=True, n="enc_conv1_1")
            enc_conv1_2 = self.conv_bn_act(enc_conv1_1, f=self.ch, k=3, s=1, p="same", act=True, n="enc_conv1_2")
            enc_pool1 = layers.maxpool(k=2, s=2, p="same", name="enc_pool1")(enc_conv1_2)

            enc_conv2_1 = self.conv_bn_act(enc_pool1, f=self.ch*2, k=3, s=1, p="same", act=True, n="enc_conv2_1")
            enc_conv2_2 = self.conv_bn_act(enc_conv2_1, f=self.ch*2, k=3, s=1, p="same", act=True, n="enc_conv2_2")
            enc_pool2 = layers.maxpool(k=2, s=2, p="same", name="enc_pool2")(enc_conv2_2)

            enc_conv3_1 = self.conv_bn_act(enc_pool2, f=self.ch*4, k=3, s=1, p="same", act=True, n="enc_conv3_1")
            enc_conv3_2 = self.conv_bn_act(enc_conv3_1, f=self.ch*4, k=3, s=1, p="same", act=True, n="enc_conv3_2")
            enc_pool3 = layers.maxpool(k=2, s=2, p="same", name="enc_pool3")(enc_conv3_2)

            enc_conv4_1 = self.conv_bn_act(enc_pool3, f=self.ch*8, k=3, s=1, p="same", act=True, n="enc_conv4_1")
            enc_conv4_2 = self.conv_bn_act(enc_conv4_1, f=self.ch*8, k=3, s=1, p="same", act=True, n="enc_conv4_2")
            enc_pool4 = layers.maxpool(k=2, s=2, p="same", name="enc_pool4")(enc_conv4_2)

            enc_conv5_1 = self.conv_bn_act(enc_pool4, f=self.ch*16, k=3, s=1, p="same", act=True, n="enc_conv5_1")
            enc_conv5_2 = self.conv_bn_act(enc_conv5_1, f=self.ch*16, k=3, s=1, p="same", act=True, n="enc_conv5_2")

            up_conv4 = self.conv_bn_act(enc_conv5_2, f=self.ch*8, k=2, s=2, p="valid", act=True, trans=True, n="up_conv4")
            concat4 = self.concat(enc_conv4_2, up_conv4, "concat4")
            dec_conv4_1 = self.conv_bn_act(concat4, f=self.ch*8, k=3, s=1, p="same", act=True, n="dec_conv4_1")
            dec_conv4_2 = self.conv_bn_act(dec_conv4_1, f=self.ch*8, k=3, s=1, p="same", act=True, n="dec_conv4_2")

            up_conv3 = self.conv_bn_act(dec_conv4_2, f=self.ch*4, k=2, s=2, p="valid", act=True, trans=True, n="up_conv3")
            concat3 = self.concat(enc_conv3_2, up_conv3, "concat3")
            dec_conv3_1 = self.conv_bn_act(concat3, f=self.ch*4, k=3, s=1, p="same", act=True, n="dec_conv3_1")
            dec_conv3_2 = self.conv_bn_act(dec_conv3_1, f=self.ch*4, k=3, s=1, p="same", act=True, n="dec_conv3_2")

            up_conv2 = self.conv_bn_act(dec_conv3_2, f=self.ch*2, k=2, s=2, p="valid", act=True, trans=True, n="up_conv2")
            concat2 = self.concat(enc_conv2_2, up_conv2, "concat2")
            dec_conv2_1 = self.conv_bn_act(concat2, f=self.ch*2, k=3, s=1, p="same", act=True, n="dec_conv2_1")
            dec_conv2_2 = self.conv_bn_act(dec_conv2_1, f=self.ch*2, k=3, s=1, p="same", act=True, n="dec_conv2_2")

            up_conv1 = self.conv_bn_act(dec_conv2_2, f=self.ch, k=2, s=2, p="valid", act=True, trans=True, n="up_conv1")
            concat1 = self.concat(enc_conv1_2, up_conv1, "concat1")
            dec_conv1_1 = self.conv_bn_act(concat1, self.ch, k=3, s=1, p="same", act=True, n="dec_conv1_1")
            dec_conv1_2 = self.conv_bn_act(dec_conv1_1, self.ch, k=3, s=1, p="same", act=True, n="dec_conv1_2")

            dec_out = self.conv_bn_act(dec_conv1_2, 1, k=1, s=1, p="same", batch=False, act=False, n="dec_out")
            dec_out = tf.sigmoid(dec_out)

            return keras.Model({"in": self.input}, {"out": dec_out}, name="model")

        elif self.mode == 1:  # Shallow Unet
            enc_conv1_1 = self.conv_bn_act(self.input, f=self.ch, k=3, s=1, p="same", act=True, n="enc_conv1_1")
            enc_conv1_2 = self.conv_bn_act(enc_conv1_1, f=self.ch, k=3, s=1, p="same", act=True, n="enc_conv1_2")
            enc_pool1 = layers.maxpool(k=2, s=2, p="same", name="enc_pool1")(enc_conv1_2)

            enc_conv2_1 = self.conv_bn_act(enc_pool1, f=self.ch*2, k=3, s=1, p="same", act=True, n="enc_conv2_1")
            enc_conv2_2 = self.conv_bn_act(enc_conv2_1, f=self.ch*2, k=3, s=1, p="same", act=True, n="enc_conv2_2")
            enc_pool2 = layers.maxpool(k=2, s=2, p="same", name="enc_pool2")(enc_conv2_2)

            enc_conv3_1 = self.conv_bn_act(enc_pool2, f=self.ch*4, k=3, s=1, p="same", act=True, n="enc_conv3_1")
            enc_conv3_2 = self.conv_bn_act(enc_conv3_1, f=self.ch*4, k=3, s=1, p="same", act=True, n="enc_conv3_2")
            enc_pool3 = layers.maxpool(k=2, s=2, p="same", name="enc_pool3")(enc_conv3_2)

            enc_conv4_1 = self.conv_bn_act(enc_pool3, f=self.ch*8, k=3, s=1, p="same", act=True, n="enc_conv4_1")
            enc_conv4_2 = self.conv_bn_act(enc_conv4_1, f=self.ch*8, k=3, s=1, p="same", act=True, n="enc_conv4_2")

            up_conv3 = self.conv_bn_act(enc_conv4_2, f=self.ch*4, k=2, s=2, p="valid", act=True, trans=True, n="up_conv3")
            concat3 = self.concat(enc_conv3_2, up_conv3, "concat3")
            dec_conv3_1 = self.conv_bn_act(concat3, f=self.ch*4, k=3, s=1, p="same", act=True, n="dec_conv3_1")
            dec_conv3_2 = self.conv_bn_act(dec_conv3_1, f=self.ch*4, k=3, s=1, p="same", act=True, n="dec_conv3_2")

            up_conv2 = self.conv_bn_act(dec_conv3_2, f=self.ch*2, k=2, s=2, p="valid", act=True, trans=True, n="up_conv2")
            concat2 = self.concat(enc_conv2_2, up_conv2, "concat2")
            dec_conv2_1 = self.conv_bn_act(concat2, f=self.ch*2, k=3, s=1, p="same", act=True, n="dec_conv2_1")
            dec_conv2_2 = self.conv_bn_act(dec_conv2_1, f=self.ch*2, k=3, s=1, p="same", act=True, n="dec_conv2_2")

            up_conv1 = self.conv_bn_act(dec_conv2_2, f=self.ch, k=2, s=2, p="valid", act=True, trans=True, n="up_conv1")
            concat1 = self.concat(enc_conv1_2, up_conv1, "concat1")
            dec_conv1_1 = self.conv_bn_act(concat1, self.ch, k=3, s=1, p="same", act=True, n="dec_conv1_1")
            dec_conv1_2 = self.conv_bn_act(dec_conv1_1, self.ch, k=3, s=1, p="same", act=True, n="dec_conv1_2")

            dec_out = self.conv_bn_act(dec_conv1_2, 1, k=1, s=1, p="same", batch=False, act=False, n="dec_out")
            dec_out = tf.sigmoid(dec_out)

            return keras.Model({"in": self.input}, {"out": dec_out}, name="model")

        elif self.mode == 2:  # ResUnet
            """ Endoder 1 """
            x = Conv2D(64, 3, padding="same", strides=1)(self.input)
            x = self.batchnorm_relu(x)
            x = Conv2D(64, 3, padding="same", strides=1)(x)
            s = Conv2D(64, 1, padding="same")(self.input)
            s1 = x + s

            """ Encoder 2, 3 """
            s2 = self.residual_block(s1, 128, strides=2)
            s3 = self.residual_block(s2, 256, strides=2)

            """ Bridge """
            b = self.residual_block(s3, 512, strides=2)

            """ Decoder 1, 2, 3 """
            x = self.decoder_block(b, s3, 256)
            x = self.decoder_block(x, s2, 128)
            x = self.decoder_block(x, s1, 64)

            """ Classifier """
            outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(x)

            """ Model """
            return keras.Model({"in": self.input}, {"out": outputs}, name="model")

        elif self.mode == 3:  # ResUnet++
            c0 = self.input
            c1 = self.stem_block(c0, self.n_filters[0], strides=1)

            ## Encoder
            c2 = self.resnet_block(c1, self.n_filters[1], strides=2)
            c3 = self.resnet_block(c2, self.n_filters[2], strides=2)
            c4 = self.resnet_block(c3, self.n_filters[3], strides=2)

            ## Bridge
            b1 = self.aspp_block(c4, self.n_filters[4])

            ## Decoder
            d1 = self.attetion_block(c3, b1)
            d1 = UpSampling2D((2, 2))(d1)
            d1 = Concatenate(axis=1)([d1, c3])
            d1 = self.resnet_block(d1, self.n_filters[3])

            d2 = self.attetion_block(c2, d1)
            d2 = UpSampling2D((2, 2))(d2)
            d2 = Concatenate(axis=1)([d2, c2])
            d2 = self.resnet_block(d2, self.n_filters[2])

            d3 = self.attetion_block(c1, d2)
            d3 = UpSampling2D((2, 2))(d3)
            d3 = Concatenate(axis=1)([d3, c1])
            d3 = self.resnet_block(d3, self.n_filters[1])

            ## output
            outputs = self.aspp_block(d3, self.n_filters[0])
            outputs = Conv2D(1, (1, 1), padding="same")(outputs)
            outputs = Activation("sigmoid")(outputs)

            ## Model
            return keras.Model({"in": self.input}, {"out": outputs}, name="model")
