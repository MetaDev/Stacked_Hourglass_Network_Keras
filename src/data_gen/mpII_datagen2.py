from data_gen.data_gen_utils import *

#compared to LSP
#neck == upper_neck
#thorax and pelvis is not in LSP

joint_names = ['r_ankle', 'r_knee', 'r_hip',
                'l_hip',  'l_knee', 'l_ankle',
                'plevis', 'thorax', 'upper_neck', 'head_top',
                'r_wrist', 'r_elbow', 'r_shoulder',
                'l_shoulder', 'l_elbow', 'l_wrist']
""""
order of joints:

Right ankle
Right knee
Right hip
Left hip
Left knee
Left ankle
Right wrist
Right elbow
Right shoulder
Left shoulder
Left elbow
Left wrist
Neck
Head top
"""
mean = np.array([0.4404, 0.4440, 0.4327], dtype=np.float)
#TODO calculate std for all images in MPII dataset, the std below is from lsp dataset
std = np.array([0.229, 0.224, 0.225])
N_JOINTS=14
import json


MSPII_LSP_index=[[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[10,6],[11,7],[12,8],[13,9],[14,10],[15,11],[8,12],[9,13]]

#re order the joints to match the LSP format
def standardize_joints(joint_list):
    new_list=np.zeros((14,3))
    for j_i in MSPII_LSP_index:
        new_list[j_i[1]]=joint_list[j_i[0]]
    return new_list



import os
class MPII_dataset(DataGen):
    image_dir, joint_file = "mpii/images", "mpii/mpii_annotations.json"

    def _load_image_joints(self):
        # load train or val annotation
        with open(self.joint_file) as anno_file:
            anno = json.load(anno_file)
            out = np.empty(len(anno), dtype=object)

            image_joints = [(_anno['img_paths'], standardize_joints(np.array(_anno['joint_self']))) for _anno in anno]
            out[:] = image_joints
        return out



import matplotlib.pyplot as plt

if __name__ == "__main__":
    image_dir, joint_file = "../../data/mpii/images", "../../data/mpii/mpii_annotations.json"
    data_set = MPII_dataset(image_dir, joint_file, [128, 128], [128, 128], 1)
    train_gen,test_gen = data_set.tt_generator(32,with_meta=True)

    for i, batch in enumerate(train_gen):
        train_in, hmaps,meta = batch
        # # irst index is for the hourglass number
        hmaps = np.array(hmaps[0])

        fig, axes = plt.subplots(nrows=8, ncols=8)

        for i,(image_in, hmap) in enumerate(zip(train_in, hmaps)):
            hmap_all=np.sum(hmap,axis=-1)
            axes.flat[i*2+1].imshow(hmap_all)
            #matplot lib can work with images in the 0-1 range but not fully normalized
            axes.flat[i*2].imshow(image_in+mean)
        plt.show()
        cv2.waitKey(0)



