import numpy as np
from skimage import io
from data_gen.data_gen_utils import get_bounding_box
from imgaug import augmenters as iaa
from data_gen.data_gen_utils import expand_bbox, flip_symmetric_keypoints


joints = ['r_ankle', 'r_knee', 'r_hip',
                'l_hip',  'l_knee', 'l_ankle',
                'plevis', 'thorax', 'upper_neck', 'head_top',
                'r_wrist', 'r_elbow', 'r_shoulder',
                'l_shoulder', 'l_elbow', 'l_wrist']
mean = np.array([0.4404, 0.4440, 0.4327], dtype=np.float)
#TODO calculate std for all images in MPII dataset, the std below is from lsp dataset
std = np.array([0.229, 0.224, 0.225])
N_JOINTS=14
import json
from imgaug import parameters as iap

def _load_image_annotation(jsonfile):
    # load train or val annotation
    with open(jsonfile) as anno_file:
        anno = json.load(anno_file)
    return anno


import os
class MSPII_dataset(object):
    def __init__(self,image_dir,joint_file,inres, outres,num_hgstack):
        #load dataset to be able to return the size
        self.anno=_load_image_annotation(joint_file)
        self.outres=outres
        self.inres=inres
        self.num_hgstack=num_hgstack
        self.image_dir=image_dir

    def get_dataset_size(self):
        return len(self.anno)

    def generator(self, batch_size, sigma=5, is_shuffle=True, with_meta=False):

        '''
        Input:  batch_size * inres  * Channel (3)
        Output: batch_size * oures  * nparts
        '''

        inres = self.inres
        outres = self.outres
        num_hgstack = self.num_hgstack
        train_input = np.zeros(shape=(batch_size, inres[0], inres[1], 3), dtype=np.float)
        gt_heatmap = np.zeros(shape=(batch_size, outres[0], outres[1], N_JOINTS), dtype=np.float)
        meta_info = []
        # create a batch of images and its heatmpas and yield it
        # while True:
        for i in range(2):
            if is_shuffle:
                np.random.shuffle(self.anno)
            for _i, _anno in enumerate(self.anno):
                batch_i = _i % batch_size

                imagefile = _anno['img_paths']
                image = io.imread(os.path.join(self.image_dir, imagefile))
                # get center
                joint_list = np.array(_anno['joint_self'])
                box= get_bounding_box(joint_list, image)

                # d_util.draw_image_with_joints(image, joints)
                image = image[box[1]:box[3], box[0]:box[2], :]
                joint_list[:, :2] = joint_list[:, :2] - np.array([box[0], box[1]])
                # d_util.draw_image_with_joints(image, joints)

                #
                # # draw_image_with_joints(image,joint_list)
                #

                # augment image data, apply 2 of the augmentations

                # TODO add other interesting augmentation as such that less keypoints are lost e.g. PerspectiveTransform
                # or contrary hide keypoints in image data with object to make model more robust

                # the augmentation doesn't take into account that flipping switches the semantic meaning of left and right
                flip_j = lambda keypoints_on_images, random_state, parents, hooks: flip_symmetric_keypoints(
                    keypoints_on_images)
                noop = lambda images, random_state, parents, hooks: images
                seq = iaa.SomeOf(5, [
                    iaa.Sometimes(0.4, iaa.Scale(iap.Uniform(0.5,1.0))),
                    iaa.Sometimes(0.6, iaa.CropAndPad(percent=(-0.25, 0.25), pad_mode=["edge"], keep_size=False)),
                    iaa.Sequential([iaa.Fliplr(0.1), iaa.Lambda(noop, flip_j)]),
                    iaa.Sometimes(0.4, iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 50))),
                    iaa.Sometimes(0.1, iaa.GaussianBlur(sigma=(0, 3.0)))
                ])
                #
                # seq_det = seq.to_deterministic()
                # image_aug = seq_det.augment_image(image)
                # # augment keyponts accordingly
                # joint_list[:, :2] = apply_iaa_keypoints(seq_det, joint_list[:, :2], image_aug.shape)
                #
                # # show the images with joints visible
                # draw_image_with_joints(image_aug, joint_list)
                #
                # # normalize image channels and scale the input image and keypoints respectively
                # img_scale = iaa.Scale({"height": inres[0], "width": inres[1]})
                # image_aug = img_scale.augment_image(image_aug)
                # kp_scale = iaa.Scale({"height": outres[0], "width": outres[1]})
                # joint_list[:, :2] = apply_iaa_keypoints(kp_scale, joint_list[:, :2], outres)
                # image_aug = ((image_aug / 255.0) - mean) / std
                #
                # train_input[batch_i, :, :, :] = image_aug
                #
                # gt_hmp = generate_gtmap(joint_list, sigma, outres)
                # gt_heatmap[batch_i, :, :, :] = gt_hmp
                # # save keypoints that created the heatmap
                # meta_info.append({'tpts': joint_list})
                # if batch_i == (batch_size - 1):
                #     out_hmaps = [gt_heatmap] * num_hgstack
                #     if not with_meta:
                #         yield train_input, out_hmaps
                #     else:
                #         yield train_input, out_hmaps, meta_info
                #         meta_info = []

if __name__ == "__main__":
    image_dir, joint_file = "../../data/mpii/images", "../../data/mpii/mpii_annotations.json"
    data_set = MSPII_dataset(image_dir, joint_file, [128, 128], [128, 128], 1)
    train_gen = data_set.generator(100)
    for i, _ in enumerate(train_gen):
        if i == 2:
            break
