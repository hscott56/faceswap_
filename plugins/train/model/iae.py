#!/usr/bin/env python3
""" Improved autoencoder for faceswap """

from keras.layers import Concatenate, Conv2D, Dense, Flatten, Input, Reshape
from keras.models import Model as KerasModel

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
        self.add_network("intermediate", "a", self.intermediate())
        self.add_network("intermediate", "b", self.intermediate())
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
            inter_side = self.networks["intermediate_{}".format(side)].network
            output = decoder(Concatenate()([inter_side(encoder(inp)),
                                            inter_both(encoder(inp))]))

            autoencoder = KerasModel(inp, output)
            self.add_predictor(side, autoencoder)
        logger.debug("Initialized model")

    def encoder(self):
        """ Encoder Network """
        input_ = Input(shape=self.input_shape)
        sizes = [self.encoder_dim // 8, self.encoder_dim // 4,
                 self.encoder_dim // 2, self.encoder_dim]
        names = ['1st_conv', '2nd_conv', '3rd_conv', '4th_conv']

        x = input_
        for size, name in zip(sizes,names):
            x = self.blocks.conv(x, size, name=name)

        x = Flatten()(x)
        return KerasModel(input_, x)

    def intermediate(self):
        """ Intermediate Network """
        latent_shape = self.input_shape[0] // 16
        input_ = Input(shape=(None, latent_shape * latent_shape * self.encoder_dim))
        x = input_
        x = Dense(self.encoder_dim, name = '1st_dense')(x)
        x = Dense(latent_shape * latent_shape * self.encoder_dim //2, name = '2nd_dense')(x)
        x = Reshape((latent_shape, latent_shape, self.encoder_dim //2))(x)
        return KerasModel(input_, x)

    def decoder(self):
        """ Decoder Network """
        input_ = Input(shape=(self.input_shape[0] // 16, self.input_shape[0] // 16, self.encoder_dim))
        sizes = [self.encoder_dim // 2, self.encoder_dim // 4,
                 self.encoder_dim // 8, self.encoder_dim // 16]
        names = ['1st_upscale', '2nd_upscale',
                 '3rd_upscale', '4th_upscale']

        x = input_
        for size, name in zip(sizes,names):
            x = self.blocks.upscale(x, size, name = name)

        x = Conv2D(3, kernel_size=5, padding="same",
                   activation="sigmoid", name = 'output_sigmoid')(x)
        return KerasModel(input_, x)
