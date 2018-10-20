from os.path import basename
from scipy.io import loadmat
import glob
import re
import os.path
ROOT_DIR="../../data/lspet/"
#for practical applications and the use of multiple dataset we only use 14 joints
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


import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

def pose2keypoints( shape, pose):
    keypoints = []
    for row in range(int(pose.shape[0])):
        x = pose[row, 0]
        y = pose[row, 1]
        keypoints.append(ia.Keypoint(x=x, y=y))
    return ia.KeypointsOnImage(keypoints, shape=shape)


def keypoints2pose(keypoints_aug):
    one_person = []
    for kp_idx, keypoint in enumerate(keypoints_aug.keypoints):
        x_new, y_new = keypoint.x, keypoint.y
        one_person.append(np.array(x_new).astype(np.float32))
        one_person.append(np.array(y_new).astype(np.float32))
    return np.array(one_person).reshape([-1, 2])


from skimage import io, transform
#create a bigger bounding box
def expand_bbox(left, right, top, bottom, img_width, img_height):
    width = right-left
    height = bottom-top
    ratio = 0.15
    new_left = np.clip(left-ratio*width,0,img_width)
    new_right = np.clip(right+ratio*width,0,img_width)
    new_top = np.clip(top-ratio*height,0,img_height)
    new_bottom = np.clip(bottom+ratio*height,0,img_height)

    return [int(new_left), int(new_top), int(new_right), int(new_bottom)]
from data_gen.data_process import generate_gtmap
import cv2
import scipy
from  data_gen.utest_data_view import draw_joints
from imgaug import parameters as iap

def apply_iaa_keypoints(iaa, keypoints, shape):
    return keypoints2pose(iaa.augment_keypoints([pose2keypoints(shape, keypoints)])[0])
def draw_image_with_joints(image,joint_list):
    test = np.copy(image).astype(np.uint8)
    draw_joints(test, joint_list)
    cv2.imshow('image', cv2.cvtColor(test, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
def set_outside_joints_invisible(joint_list,outres):
    for joint in joint_list:
        if not 0<=joint[0]<outres[0] or not 0<=joint[1]<outres[1]:
            joint[2]=0
class LSP_dataset(object):
    def __init__(self,image_dir,joint_file):
        #load dataset to be able to return the size
        self.data_img_joints=create_data(image_dir,joint_file)
    def get_dataset_size(self):
        return len(self.data_img_joints)
    def generator(self,batch_size,inres, outres,num_hgstack, sigma=5,  is_shuffle=True):
        #I don't know where these numbers come from but at least the mean comes back in several implementations of pose estimation
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        '''
        Input:  batch_size * inres  * Channel (3)
        Output: batch_size * oures  * nparts
        '''
        nparts=np.shape(self.data_img_joints[0][1])[0]
        train_input = np.zeros(shape=(batch_size, inres[0], inres[1], 3), dtype=np.float)
        gt_heatmap  = np.zeros(shape=(batch_size, outres[0], outres[1], nparts), dtype=np.float)

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

                # draw_image_with_joints(image,joint_list)

                #set bounding box
                height, width = image.shape[0], image.shape[1]

                xmin = np.min(joint_list[:, 0])
                ymin = np.min(joint_list[:, 1])
                xmax = np.max(joint_list[:, 0])
                ymax = np.max(joint_list[:, 1])
                box = expand_bbox(xmin, xmax, ymin, ymax, width, height)
                image = image[box[1]:box[3], box[0]:box[2], :]
                #change keypoints according to new bounding box
                joint_list[:,:2] = joint_list[:,:2] - np.array([box[0], box[1]])

                # draw_image_with_joints(image,joint_list)

                # augment image data, apply 2 of the augmentations
                #TODO add other interesting augmentation as such that less keypoints are lost e.g. PerspectiveTransform
                seq = iaa.SomeOf(2, [
                    iaa.Sometimes(0.4, iaa.Scale(iap.Uniform(0.5,1.0))),
                    iaa.Sometimes(0.6, iaa.CropAndPad(percent=(-0.25, 0.25), pad_mode=["edge"], keep_size=False)),
                    iaa.Fliplr(0.1),
                    iaa.Sometimes(0.4, iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 50))),
                    iaa.Sometimes(0.1, iaa.GaussianBlur(sigma=(0, 3.0)))
                ])

                seq_det = seq.to_deterministic()
                image_aug = seq_det.augment_image(image)
                # augment keyponts accordingly
                joint_list[:, :2] = apply_iaa_keypoints(seq_det,joint_list[:,:2],image_aug.shape)

                #show the images with joints visible
                # draw_image_with_joints(image_aug,joint_list)

                #normalize image channels and scale the input image and keypoints respectively
                img_scale=iaa.Scale({"height": inres[0], "width": inres[1]})
                image_aug = img_scale.augment_image(image_aug)
                kp_scale = iaa.Scale({"height": outres[0], "width": outres[1]})
                joint_list[:, :2] = apply_iaa_keypoints(kp_scale , joint_list[:, :2], outres)
                image_aug = ((image_aug / 255.0) - mean)/std

                train_input[batch_i, :, :, :] = image_aug

                gt_hmp=generate_gtmap(joint_list, sigma, outres)
                gt_heatmap[batch_i, :, :, :] = gt_hmp
                if batch_i  == (batch_size - 1):
                    out_hmaps = [gt_heatmap] * num_hgstack
                    yield train_input, out_hmaps
if __name__ == "__main__":
    image_dir, joint_file = "../../data/lspet/images", "../../data/lspet/joints.mat"
    data_set = LSP_dataset(image_dir, joint_file)
    train_gen = data_set.generator(100, [128,128], [128,128], 1)
    for i,_ in enumerate(train_gen):
        if i==2:
            break


