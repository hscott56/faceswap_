#!/usr/bin/env python3
""" DeepFakesLab H128 Model
    Based on https://github.com/iperov/DeepFaceLab
"""

from keras.layers import Conv2D, Dense, Flatten, Input, Reshape
from keras.models import Model as KerasModel

from .original import logger, Model as OriginalModel


class Model(OriginalModel):
    """ Low Memory version of Original Faceswap Model """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s",
                     self.__class__.__name__, args, kwargs)

        kwargs["input_shape"] = (128, 128, 3)
        kwargs["encoder_dim"] = 256 if self.config["lowmem"] else 512

        super().__init__(*args, **kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    def set_training_data(self):
        """ Set the dictionary for training """
        self.training_opts["mask_type"] = self.config["mask_type"]
        super().set_training_data()

    def build_autoencoders(self):
        """ Initialize DFL H128 model """
        logger.debug("Initializing model")
        mask_shape = self.input_shape[:2] + (1, )
        for side in ("a", "b"):
            face_in = Input(shape=self.input_shape, name="face")
            mask_in = Input(shape=mask_shape, name="mask")
            decoder = self.networks["decoder_{}".format(side)].network
            decoder.mask = False
            face_out = decoder(self.networks["encoder"].network(face_in))
            decoder.mask = True
            mask_out = decoder(self.networks["encoder"].network(face_in))
            autoencoder = KerasModel([face_in, mask_in], [face_out, mask_out])
            self.add_predictor(side, autoencoder)
        logger.debug("Initialized model")

    def encoder(self):
        """ DFL H128 Encoder --- this is procedureally identical to the
            stadard encoder"""
        input_ = Input(shape=self.input_shape)
        latent_shape = self.input_shape[0] // 16
        sizes = [self.encoder_dim // 8, self.encoder_dim // 4,
                 self.encoder_dim // 2, self.encoder_dim]
        names = ['1st_conv', '2nd_conv', '3rd_conv','4th_conv']

        x = input_
        for size, name in zip(sizes,names):
            x = self.blocks.conv(x, size, name=name)

        x = Flatten()(x)
        x = Dense(self.encoder_dim, name = '1st_dense')(x)
        x = Dense(latent_shape * latent_shape * self.encoder_dim, name = '2nd_dense')(x)
        x = Reshape((latent_shape, latent_shape, self.encoder_dim))(x)

        x = self.blocks.upscale(x, self.encoder_dim, name = '1st_upscale')
        return KerasModel(input_, x)

    def decoder(self):
        """ DFL H128 Decoder """
        latent_shape = self.input_shape[0] // 16
        input_ = Input(shape=(latent_shape, latent_shape, self.encoder_dim))
        sizes = [self.encoder_dim, self.encoder_dim // 2, self.encoder_dim // 4]
        names = ['2nd_upscale', '3rd_upscale', '4th_upscale']
        if self.mask:
           names = [name + '_mask' for name in names]
           channel_num = 1
           out_name = 'mask_sigmoid'
        else:
           channel_num = 3
           out_name = 'face_sigmoid'

        x = input_
        for size, name in zip(sizes,names):
            x = self.blocks.upscale(x, size, name = name)

        x = Conv2D(channel_num, kernel_size=5, padding='same',
                   activation='sigmoid', name = out_name)(x)

        return KerasModel(input_, x)
