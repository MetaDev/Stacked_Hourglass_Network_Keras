
import os
from net.hg_blocks import create_hourglass_network, bottleneck_mobile
from data_gen.mpii_datagen import MPIIDataGen
from keras.callbacks import CSVLogger
from keras.models import model_from_json

import datetime
import scipy.misc
from data_gen.data_process import normalize
import numpy as np
from eval.eval_callback import EvalCallBack
import tools.flags as fl
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


    def train(self,data_gen_class,batch_size,model_path,data_path, epochs,debug=True):
        data_set=data_gen_class(os.path.join(data_path,data_gen_class.image_dir),
                                os.path.join(data_path,data_gen_class.joint_file),
                                 self.inres, self.outres, self.num_hgstacks)
        test_fract = 0.2
        train_gen, test_gen = data_set.tt_generator(batch_size, test_portion=test_fract)
        csvlogger = CSVLogger(
            os.path.join(model_path, "csv_train_" + str(datetime.datetime.now().strftime('%d_%m-%H_%M')) + ".csv"))
        val_gen=data_set.val_generator(batch_size)
        checkpoint = EvalCallBack(model_path,self,val_gen)

        xcallbacks = [csvlogger,checkpoint]

        train_steps = (data_set.get_dataset_size() * (1 - test_fract)) // batch_size
        test_steps = (data_set.get_dataset_size() * (test_fract)) // batch_size
        #DEBUG
        print(fl.DEBUG)
        if fl.DEBUG:
            train_steps,test_steps=1,1
        self.model.fit_generator(generator=train_gen, steps_per_epoch=train_steps,
                                 validation_data=test_gen, validation_steps=test_steps,
                                 epochs=epochs, callbacks=xcallbacks)


    def train_old(self, batch_size, model_path, epochs):
        train_dataset = MPIIDataGen("../../data/mpii/mpii_annotations.json", "../../data/mpii/images",
                                      inres=self.inres,  outres=self.outres, num_hgstack=self.num_hgstacks,is_train=True)
        train_gen = train_dataset.generator(batch_size,  sigma=1, is_shuffle=True,
                                            rot_flag=True, scale_flag=True, flip_flag=True)
        csvlogger = CSVLogger(os.path.join(model_path, "csv_train_"+ str(datetime.datetime.now().strftime('%H:%M')) + ".csv"))

        # checkpoint =  EvalCallBack(model_path,self)

        xcallbacks = [csvlogger]

        self.model.fit_generator(generator=train_gen, steps_per_epoch=train_dataset.get_dataset_size()//batch_size,
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
    def inference_rgb(self, rgbdata, orgshape):
        import data_gen

        scale = (orgshape[0] * 1.0 / self.inres[0], orgshape[1] * 1.0 / self.inres[1])
        imgdata = scipy.misc.imresize(rgbdata, self.inres)
        #WARNING unchecked code
        mean = data_gen.data_gen_utils.mean

        imgdata = normalize(imgdata, mean)

        input = imgdata[np.newaxis, :, :, :]
        #WARNING this code only works of there are more than 1 stack
        out = self.model.predict(input)
        return out[-1], scale

    def inference_file(self, imgfile, mean=None):
        imgdata = scipy.misc.imread(imgfile)
        ret = self.inference_rgb(imgdata, imgdata.shape, mean)
        return ret




