#!/usr/bin/env python3
""" DFaker Model
    Based on the dfaker model: https://github.com/dfaker """


from keras.initializers import RandomNormal
from keras.layers import Input
from keras.models import Model as KerasModel

from lib.model.nn_blocks import Conv2D, res_block, upscale

from .original import logger, Model as OriginalModel


class Model(OriginalModel):
    """ Improved Autoeencoder Model """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)
        kwargs["input_shape"] = (64, 64, 3)
        kwargs["encoder_dim"] = 1024
        self.kernel_initializer = RandomNormal(0, 0.02)
        super().__init__(*args, **kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    def set_training_data(self):
        """ Set the dictionary for training """
        self.training_opts["serializer"] = self.config["alignments_format"]
        self.training_opts["mask_type"] = self.config["mask_type"]
        self.training_opts["full_face"] = True
        super().set_training_data()

    def build_autoencoders(self):
        """ Initialize Dfaker model """
        logger.debug("Initializing model")
        mask_shape = (self.input_shape[0] * 2, self.input_shape[1] * 2, 1)
        for side in ("a", "b"):
            inp = [Input(shape=self.input_shape, name="face"),
                   Input(shape=mask_shape, name="mask")]
            decoder = self.networks["decoder_{}".format(side)].network
            output = decoder(self.networks["encoder"].network(inp[0]))
            autoencoder = KerasModel(inp, output)
            self.add_predictor(side, autoencoder)
        logger.debug("Initialized model")

    def decoder(self):
        """ Decoder Network """
        use_subpixel = self.config["subpixel_upscaling"]
        latent_shape = self.input_shape[0] // 16
        input_ = Input(shape=(latent_shape*2, latent_shape*2, self.encoder_dim // 2))

        inp_x = input_
        inp_y = input_

        inp_x = upscale(inp_x, self.encoder_dim // 2, use_subpixel=use_subpixel, name = '2nd_upscale')
        inp_x = res_block(inp_x, self.encoder_dim // 2, kernel_initializer=self.kernel_initializer)
        inp_x = upscale(inp_x, self.encoder_dim // 4, use_subpixel=use_subpixel, name = '3rd_upscale')
        inp_x = res_block(inp_x, self.encoder_dim // 4, kernel_initializer=self.kernel_initializer)
        inp_x = upscale(inp_x, self.encoder_dim // 8, use_subpixel=use_subpixel, name = '4th_upscale')
        inp_x = res_block(inp_x, self.encoder_dim // 8, kernel_initializer=self.kernel_initializer)
        inp_x = upscale(inp_x, self.encoder_dim // 16, use_subpixel=use_subpixel, name = '5th_upscale')
        inp_x = Conv2D(3,
                       kernel_size=5,
                       padding='same',
                       activation='sigmoid',
                       name = 'output_sigmoid')(inp_x)

        inp_y = upscale(inp_y, self.encoder_dim // 2 , use_subpixel=use_subpixel, name = '2nd_m_upscale')
        inp_y = upscale(inp_y, self.encoder_dim // 4 , use_subpixel=use_subpixel, name = '3rd_m_upscale')
        inp_y = upscale(inp_y, self.encoder_dim // 8 , use_subpixel=use_subpixel, name = '4th_m_upscale')
        inp_y = upscale(inp_y, self.encoder_dim // 16, use_subpixel=use_subpixel, name = '5th_m_upscale')
        inp_y = Conv2D(1,
                       kernel_size=5,
                       padding='same',
                       activation='sigmoid',
                       name = 'mask_sigmoid')(inp_y)

        return KerasModel([input_], outputs=[inp_x, inp_y])
