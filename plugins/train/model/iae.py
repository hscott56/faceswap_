#!/usr/bin/env python3
""" Improved autoencoder for faceswap """

from keras.layers import Concatenate, Dense, Flatten, Input, Reshape
from keras.layers.convolutional import Conv2D
from keras.models import Model as KerasModel

from lib.model.nn_blocks import conv, upscale
from ._base import ModelBase, logger


class Model(ModelBase):
    """ Improved Autoeencoder Model """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)
        kwargs["input_shape"] = (64, 64, 3)
        kwargs["encoder_dim"] = 1024
        super().__init__(*args, **kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    def add_networks(self):
        """ Add the IAE model weights """
        logger.debug("Adding networks")
        self.add_network("encoder", None, self.encoder())
        self.add_network("decoder", None, self.decoder())
        self.add_network("inter", "a", self.intermediate())
        self.add_network("inter", "b", self.intermediate())
        self.add_network("inter", None, self.intermediate())
        logger.debug("Added networks")

    def build_autoencoders(self):
        """ Initialize IAE model """
        logger.debug("Initializing model")
        inp = Input(shape=self.input_shape, name="face")

        decoder = self.networks["decoder"].network
        encoder = self.networks["encoder"].network
        inter_both = self.networks["inter"].network
        for side in ("a", "b"):
            inter_side = self.networks["inter_{}".format(side)].network
            output = decoder(Concatenate()([inter_side(encoder(inp)),
                                            inter_both(encoder(inp))]))

            autoencoder = KerasModel(inp, output)
            self.add_predictor(side, autoencoder)
        logger.debug("Initialized model")

    def encoder(self):
        """ Encoder Network """
        input_ = Input(shape=self.input_shape)
        var_x = input_
        var_x = conv(var_x, self.encoder_dim // 8, name='1st_conv')
        var_x = conv(var_x, self.encoder_dim // 4, name='2nd_conv')
        var_x = conv(var_x, self.encoder_dim // 2, name='3rd_conv')
        var_x = conv(var_x, self.encoder_dim, name='4th_conv')
        var_x = Flatten()(var_x)
        return KerasModel(input_, var_x)

    def intermediate(self):
        """ Intermediate Network """
        latent_shape = self.input_shape[0] // 16
        input_ = Input(shape=(None, latent_shape * latent_shape * self.encoder_dim))
        var_x = input_
        var_x = Dense(self.encoder_dim, name = '1st_dense')(var_x)
        var_x = Dense(latent_shape * latent_shape * self.encoder_dim //2, name = '2nd_dense')(var_x)
        var_x = Reshape((latent_shape, latent_shape, self.encoder_dim //2))(var_x)
        return KerasModel(input_, var_x)

    def decoder(self):
        """ Decoder Network """
        subpixel = self.config["subpixel_upscaling"]
        latent_shape = self.input_shape[0] // 16
        input_ = Input(shape=(latent_shape, latent_shape, self.encoder_dim))
        var_x = input_
        var_x = upscale(var_x, self.encoder_dim // 2 , use_subpixel=subpixel, name = '1st_upscale')
        var_x = upscale(var_x, self.encoder_dim // 4 , use_subpixel=subpixel, name = '2nd_upscale')
        var_x = upscale(var_x, self.encoder_dim // 8 , use_subpixel=subpixel, name = '3rd_upscale')
        var_x = upscale(var_x, self.encoder_dim // 16, use_subpixel=subpixel, name = '4th_upscale')
        var_x = Conv2D(3, kernel_size=5, padding="same", activation="sigmoid", name = 'output_sigmoid')(var_x)
        return KerasModel(input_, var_x)
