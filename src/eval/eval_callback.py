import keras
import os
import datetime
from time import time
from data_gen.mpii_datagen import MPIIDataGen
from eval.eval_heatmap import cal_heatmap_acc
from data_gen.lsp_datgen import LSP_dataset
class EvalCallBack(keras.callbacks.Callback):

    def __init__(self, foldpath,hourglass):
        self.foldpath = foldpath
        self.hourglass=hourglass


    def get_folder_path(self):
        return self.foldpath

    def run_eval(self, epoch):
        # if dataset=="MPII":
        #     valdata = MPIIDataGen("../../data/mpii/mpii_annotations.json",
        #                           "../../data/mpii/images",
        #                       inres=(256, 256), outres=(64, 64), is_train=False)
        # else:
        #
        image_dir, joint_file = "../../data/lspet/images", "../../data/lspet/joints.mat"
        hg=self.hourglass
        valdata = LSP_dataset(image_dir, joint_file,hg.inres,hg.outres,hg.num_hgstacks)

        total_suc, total_fail = 0, 0
        #0.2 means the predicted keyponits is within 20% of the original
        threshold = 0.2

        count = 0
        batch_size = 8
        for _img, _gthmap, _meta in valdata.generator(batch_size, is_shuffle=False, with_meta=True):

            count += batch_size
            # if count > valdata.get_dataset_size():
            #WARNING this is cpu test code
            if count > 10:
                break
            out = self.model.predict(_img)
            suc, bad = cal_heatmap_acc(out[-1], _meta, threshold)

            total_suc += suc
            total_fail += bad

        acc = total_suc*1.0 / (total_fail + total_suc)

        print('Eval Accuray ', acc, '@ Epoch ', epoch)

        with open(os.path.join(self.get_folder_path(), 'val.txt'), 'a+') as xfile:
            xfile.write('Epoch ' + str(epoch) + ':' + str(acc) + '\n')

    def on_epoch_end(self, epoch, logs=None):
        # This is a walkaround to sovle model.save() issue
        # in which large network can't be saved due to size.
        
        # save model to json
        if epoch == 0:
            jsonfile = os.path.join(self.foldpath, "net_arch.json")
            with open(jsonfile, 'w') as f:
                f.write(self.model.to_json())

        # save weights
        modelName = os.path.join(self.foldpath, "weights_epoch" + str(epoch) + ".h5")
        self.model.save_weights(modelName)

        print("Saving model to ", modelName)

        #TODO for now don't evaluate on the whole dataset with this particular metric
        #self.run_eval(epoch)

