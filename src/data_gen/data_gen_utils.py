import imgaug as ia
from imgaug import parameters as iap
import numpy as np
import cv2
from imgaug import augmenters as iaa
import os
from skimage import io
from data_gen.data_process import generate_gtmap

#standard joint order
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

def image_with_joints(image,joint_list,colormap=None):
    test = np.copy(image).astype(np.uint8)
    paint_joints(test, joint_list, colormap)
    return test
def draw(image):
    cv2.imshow('image', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)


def expand_bbox(left, right, top, bottom, img_width, img_height,ratio=0.15):
    width = right-left
    height = bottom-top
    new_left = np.clip(left-ratio*width,0,img_width)
    new_right = np.clip(right+ratio*width,0,img_width)
    new_top = np.clip(top-ratio*height,0,img_height)
    new_bottom = np.clip(bottom+ratio*height,0,img_height)

    return [int(new_left), int(new_top), int(new_right), int(new_bottom)]
def draw_images(images,size=(512,512)):
    fit_im=iaa.Scale(size)
    scaled_ims= [fit_im.augment_image(im) for im in images]
    draw(np.concatenate(scaled_ims,axis=1))

def paint_joints(cvmat, joints, colormap=None):
    for j,_joint in enumerate(joints):
        _x, _y, _visibility = _joint
        if _visibility == 1.0:
            if colormap is not None:
                color=colormap[j]
            else:
                color=(255, 0, 0)
            cv2.circle(cvmat, center=(int(_x), int(_y)), color=color, radius=2,thickness=-1)


def get_bounding_box(joints,img):
    visible_joints=np.array([j for j in joints if not np.all(j==0)])
    left=np.min(visible_joints[:,0])
    top=np.min(visible_joints[:,1])
    right=np.max(visible_joints[:,0])
    bottom=np.max(visible_joints[:,1])
    height,width=img.shape[1],img.shape[0]
    return expand_bbox(left,right,top,bottom,height,width,ratio=0.4)

#flip using the LSP convention
matchedParts = (
    [0, 5],  # ankle
    [1, 4],  # knee
    [2, 3],  # hip
    [6, 11],  # wrist
    [7, 10],  # elbow
    [8, 9]  # shoulder
)
LR_colormap=np.zeros((14,3))
for p in matchedParts:
    LR_colormap[p[0]]=[255,0,0]
    LR_colormap[p[1]] = [0, 255, 0]
def flip_symmetric_keypoints(keypointsOnImage):
    keypoints=keypointsOnImage[0].keypoints
    for i, j in matchedParts:
        temp_i=keypoints[i]
        temp_j = keypoints[j]
        keypoints[j],keypoints[i] = temp_i,temp_j
    return keypointsOnImage


def keypoints2pose(keypoints_aug):
    one_person = []
    for kp_idx, keypoint in enumerate(keypoints_aug.keypoints):
        x_new, y_new = keypoint.x, keypoint.y
        one_person.append(np.array(x_new).astype(np.float32))
        one_person.append(np.array(y_new).astype(np.float32))
    return np.array(one_person).reshape([-1, 2])


def pose2keypoints( shape, pose):
    keypoints = []
    for row in range(int(pose.shape[0])):
        x = pose[row, 0]
        y = pose[row, 1]
        keypoints.append(ia.Keypoint(x=x, y=y))
    return ia.KeypointsOnImage(keypoints, shape=shape)


def apply_iaa_keypoints(iaa, keypoints, shape):
    return keypoints2pose(iaa.augment_keypoints([pose2keypoints(shape, keypoints)])[0])
#TODO calculated on whole image dataset
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


N_JOINTS=14
def n_joints_visible(joint_list):
    return len([1 for joint in joint_list if joint[2]==1])
class DataGen(object):
    def _load_image_joints(self):
        pass
    def get_dataset_size(self):
        return len(self.image_joints)
    def __init__(self,image_dir,joint_file,inres, outres,num_hgstack):
        #load dataset to be able to return the size
        self.joint_file=joint_file
        self.outres=outres
        self.inres=inres
        self.num_hgstack=num_hgstack
        self.image_dir=image_dir
        self.image_joints=self._load_image_joints()
    def tt_generator(self, batch_size,sigma=5, test_portion=0.02, is_shuffle=True,coord_regression=False, with_meta=False):
        #choose random test fraction
        test_idx=np.random.choice(np.arange(self.get_dataset_size()),int(self.get_dataset_size()*test_portion),replace=False)
        train_idx=list(set(np.arange(self.get_dataset_size()))-set(test_idx))

        train_image_joints=self.image_joints[train_idx]
        test_iamge_joints=self.image_joints[test_idx]

        train_gen=self._generator(train_image_joints, batch_size, sigma, is_shuffle=is_shuffle,
                                  coord_regression=coord_regression, with_meta=with_meta)
        test_gen=self._generator(test_iamge_joints, batch_size, sigma, is_shuffle=is_shuffle,
                                 coord_regression=coord_regression, with_meta=with_meta)
        return train_gen,test_gen
    def val_generator(self, batch_size, sigma=5):
        val_gen=self._generator(self.image_joints, batch_size, sigma, is_shuffle=False, with_meta=True)
        return val_gen
    min_visible_joints=7
    def _generator(self,image_joints, batch_size, sigma=5, is_shuffle=True,coord_regression=False, with_meta=False):

        '''
        Input:  batch_size * inres  * Channel (3)
        Output: batch_size * oures  * nparts
        '''

        inres = self.inres
        outres = self.outres
        num_hgstack = self.num_hgstack
        train_input = np.zeros(shape=(batch_size, inres[0], inres[1], 3), dtype=np.float)
        gt_heatmap = np.zeros(shape=(batch_size, outres[0], outres[1], N_JOINTS), dtype=np.float)
        gt_coord = np.zeros(shape=(batch_size, N_JOINTS *2 ), dtype=np.float)

        meta_info = []
        # create a batch of images and its heatmpas and yield it
        while True:
            if is_shuffle:
                np.random.shuffle(image_joints)
            for _i, image_joint in enumerate(image_joints):
                batch_i = _i % batch_size
                imagefile, joint_list = image_joint
                if n_joints_visible(joint_list) < self.min_visible_joints:
                    continue
                image = read_img_file(os.path.join(self.image_dir, imagefile))
                if np.any(image.shape) < 10:
                    continue
                box= get_bounding_box(joint_list, image)

                # d_util.draw_image_with_joints(image, joints)
                image = image[box[1]:box[3], box[0]:box[2], :]
                joint_list[:, :2] = joint_list[:, :2] - np.array([box[0], box[1]])

                #DEBUG
                # im_j_before=image_with_joints(image,joint_list,colormap=LR_colormap)

                # augment image data, apply 2 of the augmentations
                # the augmentation doesn't take into account that flipping switches the semantic meaning of left and right
                flip_j = lambda keypoints_on_images, random_state, parents, hooks: flip_symmetric_keypoints(
                    keypoints_on_images)
                noop = lambda images, random_state, parents, hooks: images
                seq = iaa.SomeOf(2, [
                    iaa.Sometimes(0.4, iaa.Scale(iap.Uniform(0.5,1.0))),
                    iaa.Sometimes(0.6, iaa.CropAndPad(percent=(-0.25, 0.25), pad_mode=["edge"], keep_size=False)),
                    iaa.Sometimes(0.2,iaa.Sequential([iaa.Fliplr(1), iaa.Lambda(noop, flip_j)])),
                    iaa.Sometimes(0.4, iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 50))),
                    iaa.Sometimes(0.1, iaa.GaussianBlur(sigma=(0, 3.0)))
                ])

                try:
                    seq_det = seq.to_deterministic()
                    image_aug = seq_det.augment_image(image)
                except AssertionError:
                    print("image augm fail: ",imagefile,image.shape, flush=True )
                    #if augmentation fails skip this image
                    continue
                # augment keyponts accordingly
                joint_list[:, :2] = apply_iaa_keypoints(seq_det, joint_list[:, :2], image.shape)

                # show the images with joints visible
                #DEBUG
                # im_j_after=image_with_joints(image_aug, joint_list, colormap=LR_colormap)
                # draw_images([im_j_before,im_j_after])


                # normalize image channels and scale the input image and keypoints respectively
                img_scale = iaa.Scale({"height": inres[0], "width": inres[1]})
                try:
                    image_aug = img_scale.augment_image(image_aug)
                except AssertionError:
                    print("image inres scale fail: ", img_scale , inres , image_aug.shape, flush=True)
                    # if augmentation fails skip this image
                    continue


                kp_scale = iaa.Scale({"height": outres[0], "width": outres[1]})
                joint_list[:, :2] = apply_iaa_keypoints(kp_scale, joint_list[:, :2], outres)
                image_aug = normalize_img(image_aug)

                train_input[batch_i, :, :, :] = image_aug

                gt_hmp = generate_gtmap(joint_list, sigma, outres)
                gt_heatmap[batch_i, :, :, :] = gt_hmp
                #batch, joint, coord
                #normalise joint coords
                gt_coord[batch_i, :] = (joint_list[:,:2]/np.array(outres)).flatten()
                # save keypoints that created the heatmap
                if with_meta:
                    meta_info.append({'joint_list': joint_list})
                if batch_i == (batch_size - 1):
                    if coord_regression:
                        yield train_input, gt_coord
                    else:
                        out_hmaps = [gt_heatmap] * num_hgstack
                        if not with_meta:
                            yield train_input, out_hmaps
                        else:
                            yield train_input, out_hmaps, meta_info
                            meta_info = []
def read_img_file(img_file):
    return io.imread(img_file)

def normalize_img(img_data):
    '''
    :param imgdata: image in 0 ~ 255
    :return:  image from 0.0 to 1.0
    '''

    return ((img_data / 255.0) - mean) / std