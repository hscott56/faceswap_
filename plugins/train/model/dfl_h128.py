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
            inp = [Input(shape=self.input_shape, name="face"),
                   Input(shape=mask_shape, name="mask")]
            decoder = self.networks["decoder_{}".format(side)].network
            output = decoder(self.networks["encoder"].network(inp[0]))
            autoencoder = KerasModel(inp, output)
            self.add_predictor(side, autoencoder)
        logger.debug("Initialized model")

    def encoder(self):
        """ DFL H128 Encoder --- this is procedureally identical to the
            stadard encoder"""
        input_ = Input(shape=self.input_shape)
<<<<<<< HEAD
        use_subpixel = self.config["subpixel_upscaling"]
        latent_shape = self.input_shape[0] // 16

        sizes = [self.encoder_dim // 8, self.encoder_dim // 4,
                 self.encoder_dim // 2, self.encoder_dim]
        names = ['1st_conv', '2nd_conv', '3rd_conv','4th_conv']
        
        var_x = input_
        for size, name in zip(sizes,names):
            var_x = conv(var_x, size, name=name)
            
        var_x = Flatten()(var_x)
        var_x = Dense(self.encoder_dim, name = '1st_dense')(var_x)
        var_x = Dense(latent_shape * latent_shape * self.encoder_dim, name = '2nd_dense')(var_x)
        var_x = Reshape((latent_shape, latent_shape, self.encoder_dim))(var_x)
        
        var_x = upscale(var_x, self.encoder_dim, use_subpixel=use_subpixel, name = '1st_upscale')
=======
        var_x = input_
        var_x = self.blocks.conv(var_x, 128)
        var_x = self.blocks.conv(var_x, 256)
        var_x = self.blocks.conv(var_x, 512)
        var_x = self.blocks.conv(var_x, 1024)
        var_x = Dense(self.encoder_dim)(Flatten()(var_x))
        var_x = Dense(8 * 8 * self.encoder_dim)(var_x)
        var_x = Reshape((8, 8, self.encoder_dim))(var_x)
        var_x = self.blocks.upscale(var_x, self.encoder_dim)
>>>>>>> train_refactor
        return KerasModel(input_, var_x)

    def decoder(self):
        """ DFL H128 Decoder """
<<<<<<< HEAD
        use_subpixel = self.config["subpixel_upscaling"]
        latent_shape = self.input_shape[0] // 16
        input_ = Input(shape=(latent_shape, latent_shape, self.encoder_dim))
        
        sizes = [self.encoder_dim, self.encoder_dim // 2, self.encoder_dim // 4]
        names = ['2nd_upscale', '3rd_upscale', '4th_upscale']
        names = [name + '_mask' for name in names] if mask else names
        channel_num = 1 if mask else 3
        out_name = 'face_sigmoid' if mask else 'mask_sigmoid'

        var_x = input_
        for size, name in zip(sizes,names):
            var_x = upscale(var_x, size , use_subpixel=self.config["subpixel_upscaling"], name = name)
=======
        input_ = Input(shape=(16, 16, self.encoder_dim))
        var = input_
        var = self.blocks.upscale(var, self.encoder_dim)
        var = self.blocks.upscale(var, self.encoder_dim // 2)
        var = self.blocks.upscale(var, self.encoder_dim // 4)
>>>>>>> train_refactor

        var_x = Conv2D(channel_num, kernel_size=5,
                          padding='same',
                          activation='sigmoid',
                          name = out_name)(var_x)
                          
        return KerasModel(input_, var_x)
