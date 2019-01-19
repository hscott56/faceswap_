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
        input_ = Input(shape=(self.input_shape[0] // 8, self.input_shape[0] // 8, self.encoder_dim // 2))
        
        sizes = [self.encoder_dim // 2,
                 self.encoder_dim // 4,
                 self.encoder_dim // 8,
                 self.encoder_dim // 16]
        names = [
                 '2nd_upscale',
                 '3rd_upscale',
                 '4th_upscale',
                 '5th_upscale']
        m_names = ['2nd_m_upscale',
                   '3rd_m_upscale',
                   '4th_m_upscale',
                   '5th_m_upscale']
                 
        var_x = input_
        var_y = input_
        # adds one more resblock iteration than standard dfaker
        for size, name in zip(sizes,names):
            var_x = upscale(var_x, size , use_subpixel=self.config["subpixel_upscaling"], name = name)
            var_x = res_block(var_x, size, kernel_initializer=self.kernel_initializer)
            
        for size, name in zip(sizes,names):
            var_y = upscale(var_y, size , use_subpixel=self.config["subpixel_upscaling"], name = name)
            
        var_x = Conv2D(3,
                       kernel_size=5,
                       padding='same',
                       activation='sigmoid',
                       name = 'output_sigmoid')(var_x)
        var_y = Conv2D(1,
                       kernel_size=5,
                       padding='same',
                       activation='sigmoid',
                       name = 'mask_sigmoid')(var_y)

        return KerasModel([input_], outputs=[var_x, var_y])
