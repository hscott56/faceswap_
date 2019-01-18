#!/usr/bin/env python3
""" Original Model
    Based on the original https://www.reddit.com/r/deepfakes/
    code sample + contribs """

from keras.layers import Dense, Flatten, Input, Reshape

from keras.models import Model as KerasModel

from lib.model.nn_blocks import conv, Conv2D, upscale
from ._base import ModelBase, logger


class Model(ModelBase):
    """ Original Faceswap Model """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)

        if "input_shape" not in kwargs:
            kwargs["input_shape"] = (64, 64, 3)
        if "encoder_dim" not in kwargs:
            kwargs["encoder_dim"] = 512 if self.config["lowmem"] else 1024

        super().__init__(*args, **kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    def add_networks(self):
        """ Add the original model weights """
        logger.debug("Adding networks")
        self.add_network("decoder", "a", self.decoder())
        self.add_network("decoder", "b", self.decoder())
        self.add_network("encoder", None, self.encoder())
        logger.debug("Added networks")

    def build_autoencoders(self):
        """ Initialize original model """
        logger.debug("Initializing model")
        inp = Input(shape=self.input_shape, name="face")

        for side in ("a", "b"):
            logger.debug("Adding Autoencoder. Side: %s", side)
            decoder = self.networks["decoder_{}".format(side)].network
            output = decoder(self.networks["encoder"].network(inp))
            autoencoder = KerasModel(inp, output)
            self.add_predictor(side, autoencoder)
        logger.debug("Initialized model")

    def encoder(self):
        """ Encoder Network """
        input_ = Input(shape=self.input_shape)
        use_subpixel = self.config["subpixel_upscaling"]
        latent_shape = self.input_shape[0] // 16
        
        sizes = [self.encoder_dim // 8,
                 self.encoder_dim // 4,
                 self.encoder_dim // 2]
        names = ['1st_conv',
                 '2nd_conv',
                 '3rd_conv']
        if not self.config.get("lowmem", False):
            sizes.append(self.encoder_dim)
            names.append('4th_conv')
            
        var_x = input_
        for size, name in zip(sizes,names):
            var_x = conv(var_x, size, name=name)
            
        var_x = Flatten()(var_x)
        var_x = Dense(self.encoder_dim, name = '1st_dense')(var_x)
        var_x = Dense(latent_shape * latent_shape * self.encoder_dim, name = '2nd_dense')(var_x)
        var_x = Reshape((latent_shape, latent_shape, self.encoder_dim))(var_x)
        
        var_x = upscale(var_x, self.encoder_dim // 2, use_subpixel=use_subpixel, name = '1st_upscale')
        return KerasModel(input_, var_x)

    def decoder(self):
        """ Decoder Network """
        sizes = [self.encoder_dim // 4,
                 self.encoder_dim // 8,
                 self.encoder_dim // 16]
        names = ['2nd_upscale',
                 '3rd_upscale',
                 '4th_upscale']
        
        input_ = Input(shape=(self.input_shape[0] // 8, self.input_shape[0] // 8, self.encoder_dim // 2))         
        var_x = input_
        for size, name in zip(sizes,names):
            var_x = upscale(var_x, size , use_subpixel=self.config["subpixel_upscaling"], name = name)
            
        var_x = Conv2D(3, kernel_size=5, padding="same", activation="sigmoid", name = 'output_sigmoid')(var_x)
        return KerasModel(input_, var_x)
