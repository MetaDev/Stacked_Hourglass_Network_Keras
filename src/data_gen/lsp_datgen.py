from os.path import basename
from scipy.io import loadmat
import glob
import re
import os.path

from data_gen.data_gen_utils import *

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

c_r=(255,0,0)
c_l=(0,0,255)

colormap=[c_r]*3+[c_l]*3+[c_r]*3+[c_l]*3+[(0,255,255),(255,0,255)]
matchedParts = (
    [0, 5],  # ankle
    [1, 4],  # knee
    [2, 3],  # hip
    [6, 11],  # wrist
    [7, 10],  # elbow
    [8, 9]  # shoulder
)
#for practical applications and the use of multiple dataset we only use 14 joints
#for now, it can be investigated that richer dataset can be used in such a way that for the
#datasets with less joints it is simply set to invisible
N_JOINTS=14
def create_data(images_dir, joints_mat_path, transpose_order=(2, 0, 1)):
    """
    The file joints.mat is a MATLAB data file containing the joint annotations
      in a 3x14x10000 matrix called 'joints' with x and y locations
      and a binary value indicating the visbility of each joint

    Create a list of lines in format:
      image_path, x1, y1, x2,y2, ...
      where xi, yi - coordinates of the i-th joint
    """
    joints = loadmat(joints_mat_path)
    joints = joints['joints'].transpose(*transpose_order)
    lines = list()
    for img_path in sorted(glob.glob(os.path.join(images_dir, '*.jpg'))):
        index = int(re.search(r'im([0-9]+)', basename(img_path)).groups()[0]) - 1
        out_list = [img_path,joints[index]]
        lines.append(out_list)
    return lines


from imgaug import augmenters as iaa
import numpy as np
from imgaug import parameters as iap


#TODO
#keypoints can be made from coord array: from_coords_array, get_coords_array


from skimage import io
#create a bigger bounding box
from data_gen.data_process import generate_gtmap

#calculated on whole image dataset
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

#TODO use the bounding box method from util method

class LSP_dataset(object):
    def __init__(self,image_dir,joint_file,inres, outres,num_hgstack):
        #load dataset to be able to return the size
        self.data_img_joints=create_data(image_dir,joint_file)
        self.outres=outres
        self.inres=inres
        self.num_hgstack=num_hgstack
    def get_dataset_size(self):
        return len(self.data_img_joints)
    def generator(self,batch_size, sigma=5,  is_shuffle=True,with_meta=False):

        '''
        Input:  batch_size * inres  * Channel (3)
        Output: batch_size * oures  * nparts
        '''
        inres=self.inres
        outres=self.outres
        num_hgstack=self.num_hgstack
        train_input = np.zeros(shape=(batch_size, inres[0], inres[1], 3), dtype=np.float)
        gt_heatmap  = np.zeros(shape=(batch_size, outres[0], outres[1], N_JOINTS), dtype=np.float)
        meta_info = []
        # create a batch of images and its heatmpas and yield it
        while True:
            if is_shuffle:
                np.random.shuffle(self.data_img_joints)
            for _i,img_joints in enumerate(self.data_img_joints):
                #joint list contains a list of joints in format x,y,b ; b being visibility
                img_f_name,joint_list=img_joints
                img_path = os.path.join(img_f_name)
                image = io.imread(img_path)
                batch_i=_i%batch_size

                # d_util.draw_image_with_joints(image,joint_list)

                #set bounding box
                height, width = image.shape[0], image.shape[1]
                #check what happens with invisible joints as their coordinates are 0 or -1, and could interfere
                xmin = np.min(joint_list[:, 0])
                ymin = np.min(joint_list[:, 1])
                xmax = np.max(joint_list[:, 0])
                ymax = np.max(joint_list[:, 1])
                box = expand_bbox(xmin, xmax, ymin, ymax, width, height)
                image = image[box[1]:box[3], box[0]:box[2], :]
                #change keypoints according to new bounding box
                joint_list[:,:2] = joint_list[:,:2] - np.array([box[0], box[1]])

                #d_util.draw_image_with_joints(image,joint_list)

                # augment image data, apply 2 of the augmentations

                #TODO add other interesting augmentation as such that less keypoints are lost e.g. PerspectiveTransform
                #or contrary hide keypoints in image data with object to make model more robust

                # the augmentation doesn't take into account that flipping switches the semantic meaning of left and right
                flip_j = lambda keypoints_on_images, random_state, parents, hooks : \
                    flip_symmetric_keypoints(keypoints_on_images,matchedParts)
                noop = lambda images, random_state, parents, hooks : images
                seq = iaa.SomeOf(2, [
                    iaa.Sometimes(0.4, iaa.Scale(iap.Uniform(0.5,1.0))),
                    iaa.Sometimes(0.6, iaa.CropAndPad(percent=(-0.25, 0.25), pad_mode=["edge"], keep_size=False)),
                    iaa.Sequential([iaa.Fliplr(0.1),iaa.Lambda(noop, flip_j)]),
                    iaa.Sometimes(0.4, iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 50))),
                    iaa.Sometimes(0.1, iaa.GaussianBlur(sigma=(0, 3.0)))
                ])

                seq_det = seq.to_deterministic()
                image_aug = seq_det.augment_image(image)
                # augment keyponts accordingly
                joint_list[:, :2] = apply_iaa_keypoints(seq_det, joint_list[:, :2], image_aug.shape)


                #show the images with joints visible
                draw_image_with_joints(image_aug,joint_list)

                #normalize image channels and scale the input image and keypoints respectively
                img_scale=iaa.Scale({"height": inres[0], "width": inres[1]})
                image_aug = img_scale.augment_image(image_aug)
                kp_scale = iaa.Scale({"height": outres[0], "width": outres[1]})
                joint_list[:, :2] = apply_iaa_keypoints(kp_scale, joint_list[:, :2], outres)
                image_aug = ((image_aug / 255.0) - mean)/std

                train_input[batch_i, :, :, :] = image_aug

                gt_hmp=generate_gtmap(joint_list, sigma, outres)
                gt_heatmap[batch_i, :, :, :] = gt_hmp
                #save keypoints that created the heatmap
                meta_info.append({'tpts': joint_list})
                if batch_i  == (batch_size - 1):
                    out_hmaps = [gt_heatmap] * num_hgstack
                    if not with_meta:
                        yield train_input, out_hmaps
                    else:
                        yield train_input, out_hmaps,meta_info
                        meta_info=[]
if __name__ == "__main__":
    image_dir, joint_file = "../../data/lspet/images", "../../data/lspet/joints.mat"
    data_set = LSP_dataset(image_dir, joint_file, [128,128], [128,128], 1)
    train_gen = data_set.generator(100)
    for i,_ in enumerate(train_gen):
        if i==2:
            break


