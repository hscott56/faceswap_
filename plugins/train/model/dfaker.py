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

<<<<<<< HEAD
    def decoder(self, mask=False):
        """ DFaker Decoder Network """
        use_subpixel = self.config["subpixel_upscaling"]
        input_ = Input(shape=(self.input_shape[0] // 8,
                              self.input_shape[0] // 8,
                              self.encoder_dim // 2))
        
        sizes = [self.encoder_dim // 2, self.encoder_dim // 4,
                 self.encoder_dim // 8, self.encoder_dim // 16]
        names = ['2nd_upscale', '3rd_upscale',
                 '4th_upscale', '5th_upscale']
        names = [name + '_mask' for name in names] if mask else names
        channel_num = 1 if mask else 3
        out_name = 'face_sigmoid' if mask else 'mask_sigmoid'
        
        var_x = input_
        # adds one more resblock iteration than standard dfaker
        for size, name in zip(sizes,names):
            var_x = upscale(var_x, size ,
                            use_subpixel=self.config["subpixel_upscaling"],
                            name = name)
            if not mask:
                var_x = res_block(var_x, size,
                                  kernel_initializer=self.kernel_initializer)
            
        var_x = Conv2D(channel_num, kernel_size=5, padding='same',
                       activation='sigmoid', name = out_name)(var_x)

        return KerasModel(input_, var_x)
=======
    def decoder(self):
        """ Decoder Network """
        input_ = Input(shape=(8, 8, 512))
        inp_x = input_
        inp_y = input_

        inp_x = self.blocks.upscale(inp_x, 512, res_block_follows=True)
        inp_x = self.blocks.res_block(inp_x, 512, kernel_initializer=self.kernel_initializer)
        inp_x = self.blocks.upscale(inp_x, 256, res_block_follows=True)
        inp_x = self.blocks.res_block(inp_x, 256, kernel_initializer=self.kernel_initializer)
        inp_x = self.blocks.upscale(inp_x, 128, res_block_follows=True)
        inp_x = self.blocks.res_block(inp_x, 128, kernel_initializer=self.kernel_initializer)
        inp_x = self.blocks.upscale(inp_x, 64)
        inp_x = Conv2D(3,
                       kernel_size=5,
                       padding='same',
                       activation='sigmoid')(inp_x)

        inp_y = self.blocks.upscale(inp_y, 512)
        inp_y = self.blocks.upscale(inp_y, 256)
        inp_y = self.blocks.upscale(inp_y, 128)
        inp_y = self.blocks.upscale(inp_y, 64)
        inp_y = Conv2D(1,
                       kernel_size=5,
                       padding='same',
                       activation='sigmoid')(inp_y)

        return KerasModel([input_], outputs=[inp_x, inp_y])
>>>>>>> train_refactor
