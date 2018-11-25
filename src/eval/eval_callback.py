import keras
import os
import numpy as np
from eval.heatmap_process import post_process_heatmap
import tools.flags as fl
#this data structure defines the limb which its length is used to normalise the keypoint error of the first joint in
#the tuple
#the reason to do it this way is that joints which are missing are often due to sideview images where
#either left or side of the body is missing
joint_eval_limb=[(0,1),(1,2),(2,1),(3,4),(4,3),(5,4),
                 (6,7),(7,8),(8,7),(9,10),(10,9),(11,10),(12,13),(13,12)]
import data_gen.data_gen_utils as dgu
class EvalCallBack(keras.callbacks.Callback):

    def __init__(self, foldpath,hourglass,generator):
        self.foldpath = foldpath
        self.hourglass=hourglass
        self.generator=generator


    def get_folder_path(self):
        return self.foldpath

    def run_eval(self, epoch,debug=True):
        joint_acc=[[] for i in range(dgu.N_JOINTS)]
        data_it = self.generator
        if fl.DEBUG:
            import itertools
            data_it = itertools.islice(data_it,2)
        for _imgs, _gthmaps,_metas in data_it:
            outs = self.hourglass.model.predict(_imgs)
            #the first axis is for the different hourglass outputs
            outs=outs[-1]

            for out,_meta in zip(outs,_metas):
                #only get the last outputed heatmap
                #TODO if no joint is found, 0,0 is returned, maybe penalise in specific way, also what is the third value in the post process?
                pre_kps = post_process_heatmap(out)
                gt_kps=_meta["joint_list"]
                for jl in joint_eval_limb:
                    #if the limb is visible in the ground truth, than the distance can be normalised
                    if (gt_kps[jl[0]][2] == 1 and gt_kps[jl[1]][2] == 1):
                        limb_dist=np.linalg.norm(gt_kps[jl[0]][0:2]-gt_kps[jl[1]][0:2])
                        pred_kp=np.array(pre_kps[jl[0]][0:2])*self.hourglass.output_scale
                        gt_kp=gt_kps[jl[0]][0:2]
                        joint_pred_dist=np.linalg.norm(gt_kp - pred_kp)
                        norm_dist=joint_pred_dist/limb_dist
                        joint_acc[jl[0]].append(norm_dist)

        joint_acc=[np.mean(acc) for acc in joint_acc]
        print('Eval Accuray ', joint_acc, '@ Epoch ', epoch)
        #
        with open(os.path.join(self.get_folder_path(), 'val.txt'), 'a+') as xfile:
            xfile.write('Epoch ' + str(epoch) + ':' + str(joint_acc) + '\n')
        pass

    def on_epoch_end(self, epoch, logs=None):
        # This is a walk-around to solve model.save() issue
        # in which large network can't be saved due to size.
        
        # save model to json
        if epoch == 0:
            jsonfile = os.path.join(self.foldpath, "net_arch.json")
            with open(jsonfile, 'w') as f:
                f.write(self.hourglass.model.to_json())

        # save weights
        modelName = os.path.join(self.foldpath, "weights_epoch" + str(epoch) + ".h5")
        self.hourglass.model.save_weights(modelName)

        print("Saving model to ", modelName)

        #TODO for now don't evaluate on the whole dataset with this particular metric
        self.run_eval(epoch)

