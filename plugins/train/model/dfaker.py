#!/usr/bin/env python3
""" DFaker Model
    Based on the dfaker model: https://github.com/dfaker """


from keras.initializers import RandomNormal
from keras.layers import Conv2D, Input
from keras.models import Model as KerasModel

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
            face_in = Input(shape=self.input_shape, name="face")
            mask_in = Input(shape=mask_shape, name="mask")
            decoder = self.networks["decoder_{}".format(side)].network
            face_out = decoder(self.networks["encoder"].network(inp[0]), mask=False)
            mask_out = decoder(self.networks["encoder"].network(inp[0]), mask=True)
            autoencoder = KerasModel([face_in, mask_in], [face_out, mask_out])
            self.add_predictor(side, autoencoder)
        logger.debug("Initialized model")

    def decoder(self, mask=False):
        """ DFaker Decoder Network """
        input_ = Input(shape=(self.input_shape[0] // 8,
                              self.input_shape[0] // 8,
                              self.encoder_dim // 2))
        sizes = [self.encoder_dim // 2, self.encoder_dim // 4,
                 self.encoder_dim // 8, self.encoder_dim // 16]
        names = ['2nd_upscale', '3rd_upscale',
                 '4th_upscale', '5th_upscale']
        if mask:
            names = [name + '_mask' for name in names]
            channel_num = 1
            out_name = 'mask_sigmoid'
        else:
            channel_num = 3
            out_name = 'face_sigmoid'

        x = input_
        # adds one more resblock iteration than standard dfaker
        for size, name in zip(sizes,names):
            x = self.blocks.upscale(x, size, res_block_follows=True, name = name)
            if not mask:
                x = self.blocks.res_block(x, size, kernel_initializer=self.kernel_initializer)

        x = Conv2D(channel_num, kernel_size=5, padding='same',
                       activation='sigmoid', name = out_name)(x)

        return KerasModel(input_, x)
