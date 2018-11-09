
import os
from net.hg_blocks import create_hourglass_network, euclidean_loss, bottleneck_block, bottleneck_mobile
from data_gen.mpii_datagen import MPIIDataGen
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.models import load_model, model_from_json
from keras.optimizers import Adam, RMSprop
from keras.losses import mean_squared_error
import datetime
import scipy.misc
from data_gen.data_process import normalize
import numpy as np
from eval.eval_callback import EvalCallBack

class HourglassNet(object):
    # def get_output(self):
    def __init__(self, num_classes, num_hgstacks, inres, outres):
        self.num_classes = num_classes
        self.num_hgstacks = num_hgstacks
        self.inres = inres
        self.outres = outres


    def build_model(self, mobile=False, show=False):
        if mobile:
            self.model = create_hourglass_network(self.num_classes, self.num_hgstacks, self.inres, self.outres, bottleneck_mobile)
        else:
            self.model = create_hourglass_network(self.num_classes, self.num_hgstacks, self.inres, self.outres, bottleneck_block)
        # show model summary and layer name
        if show :
            self.model.summary()
    def train_lsp(self,batch_size,model_path,data_path, epochs):
        import data_gen.lsp_datgen as lsp
        image_dir, joint_file = "lspet/images", "lspet/joints.mat"
        data_set=lsp.LSP_dataset(os.path.join(data_path,image_dir), os.path.join(data_path,joint_file),
                                 self.inres, self.outres, self.num_hgstacks)
        train_gen = data_set.generator(batch_size)
        csvlogger = CSVLogger(
            os.path.join(model_path, "csv_train_" + str(datetime.datetime.now().strftime('%d-%m;%H:%M')) + ".csv"))

        checkpoint = EvalCallBack(model_path,self)

        xcallbacks = [csvlogger,checkpoint]
        steps=data_set.get_dataset_size() // batch_size
        self.model.fit_generator(generator=train_gen, steps_per_epoch=steps,
                                 # validation_data=val_gen, validation_steps= val_dataset.get_dataset_size()//batch_size,
                                 epochs=epochs, callbacks=xcallbacks)

    def trainMPII2(self, batch_size, model_path, epochs):
        train_dataset = MPIIDataGen("../../data/mpii/mpii_annotations.json", "../../data/mpii/images",
                                    inres=self.inres, outres=self.outres, is_train=True)
        train_gen = train_dataset.generator(batch_size, self.num_hgstacks, sigma=1, is_shuffle=True,
                                            rot_flag=True, scale_flag=True, flip_flag=True)
        print(os.path.join(model_path, "csv_train_" + str(datetime.datetime.now().strftime('%H:%M')) + ".csv"))
        csvlogger = CSVLogger(
            os.path.join(model_path, "csv_train_" + str(datetime.datetime.now().strftime('%H:%M')) + ".csv"))

        checkpoint = EvalCallBack(model_path,self)

        xcallbacks = [csvlogger, checkpoint]

        self.model.fit_generator(generator=train_gen, steps_per_epoch=train_dataset.get_dataset_size() // batch_size,
                                 # validation_data=val_gen, validation_steps= val_dataset.get_dataset_size()//batch_size,
                                 epochs=epochs, callbacks=xcallbacks)
    def train(self, batch_size, model_path, epochs):
        train_dataset = MPIIDataGen("../../data/mpii/mpii_annotations.json", "../../data/mpii/images",
                                      inres=self.inres,  outres=self.outres, num_hgstack=self.num_hgstacks,is_train=True)
        train_gen = train_dataset.generator(batch_size,  sigma=1, is_shuffle=True,
                                            rot_flag=True, scale_flag=True, flip_flag=True)
        print(os.path.join(model_path, "csv_train_"+ str(datetime.datetime.now().strftime('%H:%M')) + ".csv"))
        csvlogger = CSVLogger(os.path.join(model_path, "csv_train_"+ str(datetime.datetime.now().strftime('%H:%M')) + ".csv"))

        checkpoint =  EvalCallBack(model_path,self)

        xcallbacks = [csvlogger, checkpoint]

        self.model.fit_generator(generator=train_gen, steps_per_epoch=train_dataset.get_dataset_size()//batch_size,
                                 #validation_data=val_gen, validation_steps= val_dataset.get_dataset_size()//batch_size,
                                 epochs=epochs, callbacks=xcallbacks)

    def resume_train(self, batch_size, model_json, model_weights, init_epoch, epochs):

        self.load_model(model_json, model_weights)
        self.model.compile(optimizer=RMSprop(lr=5e-4), loss=mean_squared_error, metrics=["accuracy"])

        train_dataset = MPIIDataGen("../../data/mpii/mpii_annotations.json", "../../data/mpii/images",
                                    inres=self.inres, outres=self.outres, is_train=True)

        train_gen = train_dataset.generator(batch_size, self.num_hgstacks, sigma=1, is_shuffle=True,
                                            rot_flag=True, scale_flag=True, flip_flag=True)

        model_dir = os.path.dirname(os.path.abspath(model_json))
        print(model_dir , model_json)
        csvlogger = CSVLogger(os.path.join(model_dir, "csv_train_" + str(datetime.datetime.now().strftime('%H:%M')) + ".csv"))

        checkpoint = EvalCallBack(model_dir)

        xcallbacks = [csvlogger, checkpoint]

        self.model.fit_generator(generator=train_gen, steps_per_epoch=train_dataset.get_dataset_size() // batch_size,
                                 initial_epoch=init_epoch, epochs=epochs, callbacks=xcallbacks)


    def load_model(self, modeljson, modelfile):
        with open(modeljson) as f :
            self.model = model_from_json(f.read())
        self.model.load_weights(modelfile)

    '''
    def load_model(self, modelfile):
            self.model = load_model(modelfile, custom_objects={'euclidean_loss': euclidean_loss})
    '''

    def inference_rgb(self, rgbdata, orgshape, mean=None):

        scale = (orgshape[0] * 1.0 / self.inres[0], orgshape[1] * 1.0 / self.inres[1])
        imgdata = scipy.misc.imresize(rgbdata, self.inres)

        if mean is None:
            mean = np.array([0.4404, 0.4440, 0.4327], dtype=np.float)

        imgdata = normalize(imgdata, mean)

        input = imgdata[np.newaxis, :, :, :]
        #WARNING this code only works of there are more than 1 stacks
        out = self.model.predict(input)
        return out[-1], scale

    def inference_file(self, imgfile, mean=None):
        imgdata = scipy.misc.imread(imgfile)
        ret = self.inference_rgb(imgdata, imgdata.shape, mean)
        return ret




