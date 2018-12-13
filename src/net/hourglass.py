
import os
from net.hg_blocks import create_hourglass_network, bottleneck_mobile
from data_gen.mpii_datagen import MPIIDataGen
from keras.callbacks import CSVLogger
from keras.models import model_from_json

import datetime
import scipy.misc
import keras
import numpy as np
from eval.train_callback import EvalCallBack, SaveCallBack
import tools.flags as fl
import data_gen.data_gen_utils as dg
from imgaug import augmenters as iaa
class HourglassNet(object):
    output_scale = 4
    # def get_output(self):
    def __init__(self, num_classes, num_hgstacks, inres, outres):
        self.num_classes = num_classes
        self.num_hgstacks = num_hgstacks
        self.inres = inres
        self.outres = outres


    def build_model(self, mobile="v1"):
        self.model = create_hourglass_network(self.num_classes, self.num_hgstacks, self.inres, self.outres, bottleneck_mobile)
        print("Model nr. params: ", self.model.count_params())


    def train(self,data_gen_class,batch_size,model_path,data_path, epochs,debug=True):
        data_set=data_gen_class(os.path.join(data_path,data_gen_class.image_dir),
                                os.path.join(data_path,data_gen_class.joint_file),
                                 self.inres, self.outres, self.num_hgstacks)
        test_fract = 0.2
        train_gen, test_gen = data_set.tt_generator(batch_size, test_portion=test_fract)
        timestamp= str(datetime.datetime.now().strftime('%d_%m-%H_%M'))
        csvlogger = CSVLogger(
            os.path.join(model_path, "csv_train_" + timestamp+ ".csv"))
        val_gen=data_set.val_generator(batch_size)
        model_logger = SaveCallBack(model_path,self)
        early_stop=keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=1, mode='auto')
        def lr_scheduler(epoch, lr):
            decay_rate = 0.1
            decay_step = 90
            if epoch % decay_step == 0 and epoch:
                return lr * decay_rate
            return lr
        learning_rate_sched=keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                              factor=0.5, patience=3, verbose=0, mode='auto',
                                                              min_delta=0.001, cooldown=0, min_lr=0)
        #you can run tensorboard from notebooks with tensorboard notebook package
        tensorboard=keras.callbacks.TensorBoard(log_dir='./log/'+timestamp, histogram_freq=0,
                                    write_graph=True, write_images=True)


        eval_logger = EvalCallBack(model_path,self,val_gen)

        xcallbacks = [csvlogger,model_logger,early_stop,learning_rate_sched,tensorboard]

        train_steps = (data_set.get_dataset_size() * (1 - test_fract)) // batch_size
        test_steps = (data_set.get_dataset_size() * (test_fract)) // batch_size
        #DEBUG
        if fl.DEBUG:
            train_steps,test_steps=10,10
            epochs=10
        self.model.fit_generator(generator=train_gen, steps_per_epoch=train_steps,
                                 validation_data=test_gen, validation_steps=test_steps,
                                 epochs=epochs, callbacks=xcallbacks)


    def train_old(self, batch_size, model_path, epochs):
        data_set = MPIIDataGen("../../data/mpii/mpii_annotations.json", "../../data/mpii/images",
                                      inres=self.inres,  outres=self.outres, num_hgstack=self.num_hgstacks,is_train=True)
        train_gen = data_set.generator(batch_size,  sigma=1, is_shuffle=True,
                                            rot_flag=True, scale_flag=True, flip_flag=True)
        csvlogger = CSVLogger(os.path.join(model_path, "csv_train_"+ str(datetime.datetime.now().strftime('%H:%M')) + ".csv"))

        model_logger = SaveCallBack(model_path, self)

        xcallbacks = [csvlogger, model_logger]
        train_steps = data_set.get_dataset_size()  // batch_size

        # DEBUG
        if fl.DEBUG:
            train_steps, test_steps = 1, 1

        self.model.fit_generator(generator=train_gen, steps_per_epoch=train_steps,
                                 #validation_data=val_gen, validation_steps= val_dataset.get_dataset_size()//batch_size,
                                 epochs=epochs, callbacks=xcallbacks)



    def load_model(self, modeljson, modelfile):
        with open(modeljson) as f :
            self.model = model_from_json(f.read())
        self.model.load_weights(modelfile)

    '''
    def load_model(self, modelfile):
            self.model = load_model(modelfile, custom_objects={'euclidean_loss': euclidean_loss})
    '''

    def inference_rgb(self, img_data, orgshape):
        scale = (orgshape[0] * 1.0 / self.inres[0], orgshape[1] * 1.0 / self.inres[1])

        img_scale = iaa.Scale({"height": self.inres[0], "width": self.inres[1]})

        img_data = img_scale.augment_image(img_data)
        img_data = dg.normalize_img(img_data)

        input = img_data[np.newaxis, :, :, :]
        #WARNING this code only works of there are more than 1 stack
        out = self.model.predict(input)
        if self.num_hgstacks > 1:
            out=out[-1]
        return out, scale

    def inference_file(self, imgfile):
        imgdata = dg.read_img_file(imgfile)
        ret = self.inference_rgb(imgdata, imgdata.shape)
        return ret




