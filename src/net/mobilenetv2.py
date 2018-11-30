"""MobileNet v2 models for Keras.

# Reference
- [Inverted Residuals and Linear Bottlenecks Mobile Networks for
   Classification, Detection and Segmentation]
   (https://arxiv.org/abs/1801.04381)
"""


from keras.models import Model
from keras.layers import Input, Conv2D, AveragePooling2D, Dropout
from keras.layers import Activation, BatchNormalization, add, Reshape, Dense

from keras.layers import DepthwiseConv2D

from keras.optimizers import SGD
from keras import backend as K
from keras.losses import mean_squared_error
import os
from keras.callbacks import CSVLogger
import datetime
from eval.train_callback import EvalCallBack
import tools.flags as fl
from keras.backend import int_shape
relu6 = lambda x : K.relu(x,max_value=6)

def _conv_block(inputs, filters, kernel, strides):
    """Convolution Block
    This function defines a 2D convolution operation with BN and relu6.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.

    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    return Activation(relu6)(x)


def _bottleneck(inputs, filters, kernel, t, s, r=False):
    """Bottleneck
    This function defines a basic bottleneck structure.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        r: Boolean, Whether to use the residuals.

    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    tchannel = K.int_shape(inputs)[channel_axis] * t

    x = _conv_block(inputs, tchannel, (1, 1), (1, 1))

    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(relu6)(x)

    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    if r:
        x = add([x, inputs])
    return x


def _inverted_residual_block(inputs, filters, kernel, t, strides, n):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.
    # Returns
        Output tensor.
    """

    x = _bottleneck(inputs, filters, kernel, t, strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, 1, True)

    return x

class MobileNetV2(object):
    def __init__(self, num_classes, inres):
        self.num_classes = num_classes

        self.inres = inres
        self.outres=inres
        self.num_hgstacks=1

    def build_model(self):
        input_shape=(*self.inres,3)
        inputs = Input(shape=input_shape)
        k=self.num_classes*2
        x = _conv_block(inputs, 32, (3, 3), strides=(2, 2))

        x = _inverted_residual_block(x, 16, (3, 3), t=1, strides=1, n=1)
        x = _inverted_residual_block(x, 24, (3, 3), t=6, strides=2, n=2)
        x = _inverted_residual_block(x, 32, (3, 3), t=6, strides=2, n=3)
        x = _inverted_residual_block(x, 64, (3, 3), t=6, strides=2, n=4)
        x = _inverted_residual_block(x, 96, (3, 3), t=6, strides=1, n=3)
        x = _inverted_residual_block(x, 160, (3, 3), t=6, strides=2, n=3)
        x = _inverted_residual_block(x, 320, (3, 3), t=6, strides=1, n=1)

        x = _conv_block(x, 1280, (1, 1), strides=(1, 1))
        x = AveragePooling2D(input_shape[0]//32)(x)
        x = Reshape((1, 1, 1280))(x)
        x = Dropout(0.3, name='Dropout')(x)
        x = Dense(k)(x)
        # x = Conv2D(k, (1, 1), padding='same')(x)
        output = Reshape((k,))(x)
        model = Model(inputs, output)
        rms = SGD(lr=1e-05,momentum=0.9)
        model.compile(optimizer=rms, loss=mean_squared_error, metrics=["accuracy"])
        print("Model nr. params: ", model.count_params())

        self.model=model
    def train(self,data_gen_class,batch_size,model_path,data_path, epochs):
        data_set=data_gen_class(os.path.join(data_path,data_gen_class.image_dir),
                                os.path.join(data_path,data_gen_class.joint_file),
                                 self.inres, self.outres, self.num_hgstacks)
        test_fract = 0.2
        train_gen, test_gen = data_set.tt_generator(batch_size, test_portion=test_fract,coord_regression=True)
        csvlogger = CSVLogger(
            os.path.join(model_path, "csv_train_" + str(datetime.datetime.now().strftime('%d_%m-%H_%M')) + ".csv"))
        val_gen=data_set.val_generator(batch_size)
        checkpoint = EvalCallBack(model_path,self,val_gen)

        xcallbacks = [csvlogger]

        train_steps = (data_set.get_dataset_size() * (1 - test_fract)) // batch_size
        test_steps = (data_set.get_dataset_size() * (test_fract)) // batch_size
        #DEBUG
        if fl.DEBUG:
            train_steps,test_steps=30,1
        self.model.fit_generator(generator=train_gen, steps_per_epoch=train_steps,
                                 validation_data=test_gen, validation_steps=test_steps,
                                 epochs=epochs, callbacks=xcallbacks)

