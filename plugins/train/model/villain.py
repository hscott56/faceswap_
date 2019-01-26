#!/usr/bin/env python3
""" Original - VillainGuy model
    Based on the original https://www.reddit.com/r/deepfakes/ code sample + contribs
    Adapted from a model by VillainGuy (https://github.com/VillainGuy) """

from keras.initializers import RandomNormal
from keras.layers import add, Conv2D, Dense, Flatten, Input, Reshape
from keras.models import Model as KerasModel

from lib.model.layers import PixelShuffler
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
        in_conv_filters = self.input_shape[0] + max(0,self.input_shape[0] - 128) // 4
        latent_shape = self.input_shape[0] // 16
        res_cycles = 8 if self.config.get("lowmem", False) else 16

        x = self.blocks.conv(input_, in_conv_filters, res_block_follows=True, **kwargs)
        shortcut = x
        for _ in range(res_cycles):
            x = self.blocks.res_block(x, 128, **kwargs)
            
        x = add([x, shortcut]) # consider adding scale before this layer to scale the residual chain
        x = self.blocks.conv(x, 128, **kwargs)
        x = PixelShuffler()(x)
        x = self.blocks.conv(x, 128, **kwargs)
        x = PixelShuffler()(x)
        x = self.blocks.conv(x, 128, **kwargs)
        x = self.blocks.conv_sep(x, 256, **kwargs)
        x = self.blocks.conv(x, 512, **kwargs)
        if not self.config.get("lowmem", False):
            x = self.blocks.conv_sep(x, 1024, **kwargs)
            
        x = Flatten()(x)
        x = Dense(self.encoder_dim, name = '1st_dense', **kwargs)(x)
        x = Dense(latent_shape * latent_shape * self.encoder_dim, name = '2nd_dense', **kwargs)(x)
        x = Reshape((latent_shape, latent_shape, self.encoder_dim))(x)
        
        x = self.blocks.upscale(x, self.encoder_dim // 2, name = '1st_upscale', **kwargs)
        
        return KerasModel(input_, x)
        
    def decoder(self):
        """ Decoder Network """
        latent_shape = self.input_shape[0] // 16
        kwargs = {"kernel_initializer": self.kernel_initializer}
        input_ = Input(shape=(latent_shape*2, latent_shape*2, self.encoder_dim // 2))
        sizes = [self.encoder_dim // 2, self.encoder_dim // 4, self.encoder_dim // 8]
        names = ['2nd_upscale', '3rd_upscale', '4th_upscale']
        
        x = input_
        for size, name in zip(sizes,names):
            x = self.blocks.upscale(x, size, res_block_follows=True, name = name, **kwargs)
            x = self.blocks.res_block(x, size, **kwargs)
            
        x = Conv2D(3, kernel_size=5, padding='same',
                   activation='sigmoid', name = 'output_sigmoid')(x)
                   
        return KerasModel(input_, x)
