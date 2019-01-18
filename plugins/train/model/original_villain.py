#!/usr/bin/env python3
""" Original - VillainGuy model
    Based on the original https://www.reddit.com/r/deepfakes/ code sample + contribs
    Adapted from a model by VillainGuy (https://github.com/VillainGuy) """

from keras.initializers import RandomNormal
from keras.layers import add, Conv2D, Dense, Flatten, Input, Reshape
from keras.models import Model as KerasModel

from lib.model.nn_blocks import conv, conv_sep, PixelShuffler, res_block, upscale
from .original import logger, Model as OriginalModel


class Model(OriginalModel):
    """ Original HiRes Faceswap Model """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)

        kwargs["input_shape"] = (128, 128, 3)
        kwargs["encoder_dim"] = 512 if self.config["lowmem"] else 1024
        self.kernel_initializer = RandomNormal(0, 0.02)

        super().__init__(*args, **kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)
        
    def encoder(self):
        """ Encoder Network """
        kwargs = {"kernel_initializer": self.kernel_initializer}
        input_ = Input(shape=self.input_shape)
        in_conv_filters = self.input_shape[0]
        if self.input_shape[0] > 128:
            in_conv_filters = 128 + (self.input_shape[0] - 128) // 4
        latent_shape = self.input_shape[0] // 16

        var_x = conv(input_, in_conv_filters, **kwargs)
        tmp_x = var_x
        res_cycles = 8 if self.config.get("lowmem", False) else 16
        for _ in range(res_cycles):
            var_x = res_block(var_x, self.encoder_dim // 8, **kwargs)
            
        var_x = add([var_x, tmp_x])
        var_x = conv(var_x, self.encoder_dim // 8, **kwargs)
        var_x = PixelShuffler()(var_x)
        var_x = conv(var_x, self.encoder_dim // 8, **kwargs)
        var_x = PixelShuffler()(var_x)
        var_x = conv(var_x, self.encoder_dim // 8, **kwargs)
        var_x = conv_sep(var_x, self.encoder_dim // 4, **kwargs)
        var_x = conv(var_x, self.encoder_dim // 2, **kwargs)
        if not self.config.get("lowmem", False):
            var_x = conv_sep(var_x, self.encoder_dim, **kwargs)
        
        var_x = Flatten()(var_x)
        var_x = Dense(self.encoder_dim, **kwargs)(var_x)
        var_x = Dense(latent_shape * latent_shape * self.encoder_dim, **kwargs)(var_x)
        var_x = Reshape((latent_shape, latent_shape, self.encoder_dim))(var_x)
        var_x = upscale(var_x, self.encoder_dim // 2, **kwargs)
        return KerasModel(input_, var_x)

    def decoder(self):
        """ Decoder Network """
        latent_shape = self.input_shape[0] // 16
        kwargs = {"kernel_initializer": self.kernel_initializer}
        input_ = Input(shape=(latent_shape*2, latent_shape*2, self.encoder_dim // 2))

        var_x = input_
        var_x = upscale(var_x, self.encoder_dim // 2, **kwargs)
        var_x = res_block(var_x, self.encoder_dim // 2, **kwargs)
        var_x = upscale(var_x, self.encoder_dim // 4, **kwargs)
        var_x = res_block(var_x, self.encoder_dim // 4, **kwargs)
        var_x = upscale(var_x, self.encoder_dim // 8, **kwargs)
        var_x = res_block(var_x, self.encoder_dim // 8, **kwargs)
        var_x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(var_x)
        return KerasModel(input_, var_x)
