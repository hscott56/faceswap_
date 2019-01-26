#!/usr/bin/env python3
""" Original Model
    Based on the original https://www.reddit.com/r/deepfakes/
    code sample + contribs """

from keras.layers import Conv2D, Dense, Flatten, Input, Reshape

from keras.models import Model as KerasModel

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
<<<<<<< HEAD
        latent_shape = self.input_shape[0] // 16
        
        sizes = [self.encoder_dim // 8, self.encoder_dim // 4,
                 self.encoder_dim // 2, self.encoder_dim]
        names = ['1st_conv', '2nd_conv', '3rd_conv','4th_conv']
        
        if not self.config.get("lowmem", False):
            sizes = sizes[:-1]
            names = names[:-1]
            
        var_x = input_
        for size, name in zip(sizes,names):
            var_x = conv(var_x, size, name=name)
            
        var_x = Flatten()(var_x)
        var_x = Dense(self.encoder_dim, name = '1st_dense')(var_x)
        var_x = Dense(latent_shape * latent_shape * self.encoder_dim, name = '2nd_dense')(var_x)
        var_x = Reshape((latent_shape, latent_shape, self.encoder_dim))(var_x)
        
        var_x = upscale(var_x, self.encoder_dim // 2, use_subpixel=self.config["subpixel_upscaling"], name = '1st_upscale')
=======
        var_x = input_
        var_x = self.blocks.conv(var_x, 128)
        var_x = self.blocks.conv(var_x, 256)
        var_x = self.blocks.conv(var_x, 512)
        if not self.config.get("lowmem", False):
            var_x = self.blocks.conv(var_x, 1024)
        var_x = Dense(self.encoder_dim)(Flatten()(var_x))
        var_x = Dense(4 * 4 * 1024)(var_x)
        var_x = Reshape((4, 4, 1024))(var_x)
        var_x = self.blocks.upscale(var_x, 512)
>>>>>>> train_refactor
        return KerasModel(input_, var_x)

    def decoder(self):
        """ Decoder Network """
<<<<<<< HEAD
        input_ = Input(shape=(self.input_shape[0] // 8,
                              self.input_shape[0] // 8,
                              self.encoder_dim // 2))
        
        sizes = [self.encoder_dim // 4, self.encoder_dim // 8, self.encoder_dim // 16]
        names = ['2nd_upscale', '3rd_upscale', '4th_upscale']
                 
        var_x = input_
        for size, name in zip(sizes,names):
            var_x = upscale(var_x, size , use_subpixel=self.config["subpixel_upscaling"], name = name)
            
        var_x = Conv2D(3, kernel_size=5, padding="same", activation="sigmoid", name = 'output_sigmoid')(var_x)
=======
        input_ = Input(shape=(8, 8, 512))
        var_x = input_
        var_x = self.blocks.upscale(var_x, 256)
        var_x = self.blocks.upscale(var_x, 128)
        var_x = self.blocks.upscale(var_x, 64)
        var_x = Conv2D(3, kernel_size=5, padding="same", activation="sigmoid")(var_x)
>>>>>>> train_refactor
        return KerasModel(input_, var_x)
