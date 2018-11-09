import os
import json
import scipy
import scipy.misc
from  data_gen import data_process
import numpy as np
import cv2

# Most of functions in this file are adpoted from https://github.com/bearpaw/pytorch-pose
# with minor changes to fit Keras
from data_gen.data_gen_utils import paint_joints


def load_sample_ids(jsonfile, is_train):
    # create train/val split
    with open(jsonfile) as anno_file:
        anno = json.load(anno_file)

    val_anno, train_anno = [], []
    for idx, val in enumerate(anno):
        if val['isValidation'] == True:
            val_anno.append(anno[idx])
        else:
            train_anno.append(anno[idx])

    if is_train:
        return train_anno
    else:
        return val_anno


def generate_gt_map(joints, sigma, outres):
    npart = joints.shape[0]
    gtmap = np.zeros(shape=(outres[0], outres[1], npart), dtype=float)
    for i in range(npart):
        gtmap[:, :, i] = data_process.draw_labelmap(gtmap[:,:,i], joints[i,:], sigma)
    return gtmap

def view_crop_image(anno):

    print(list(anno.keys()))
    img_paths = anno['img_paths']
    img_width = anno['img_width']
    img_height = anno['img_height']

    imgdata = scipy.misc.imread(os.path.join("../../data/mpii/images", img_paths))
    paint_joints(imgdata, anno['joint_self'])

    center = np.array(anno['objpos'])
    outimg = data_process.crop(imgdata, center= center,  scale=anno['scale_provided'], res=(256, 256), rot=0)

    print(outimg.shape)

    newjoints = data_process.transform_kp(np.array(anno['joint_self']), center, anno['scale_provided'], (64, 64), rot=0)

    # meta info
    metainfo = []
    orgjoints = np.array(anno['joint_self'])
    for i in range(newjoints.shape[0]):
        meta = {'center': center, 'scale': anno['scale_provided'], 'pts': orgjoints[i], 'tpts': newjoints[i]}
        metainfo.append(meta)

    # transform back
    tpbpts = list()
    for i in range(newjoints.shape[0]):
        tpts = newjoints[i]
        meta = metainfo[i]
        orgpts = tpts
        orgpts[0:2] = data_process.transform(tpts, meta['center'], meta['scale'], res=[64, 64], invert=1, rot=0)
        tpbpts.append(orgpts)

    print(tpbpts)

    paint_joints(imgdata, np.array(tpbpts))
    cv2.imshow('image', cv2.cvtColor(imgdata, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)


def main():
    annolst = load_sample_ids("../../data/mpii/mpii_annotations.json", is_train=False)
    for _anno in annolst:
        view_crop_image(_anno)


if __name__ == '__main__':
    main()