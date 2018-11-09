import imgaug as ia
import numpy as np
import cv2

def draw_image_with_joints(image,joint_list,colormap=None):
    test = np.copy(image).astype(np.uint8)
    draw_joints(test, joint_list,colormap)
    cv2.imshow('image', cv2.cvtColor(test, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
def draw_image(image):
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


def draw_joints(cvmat, joints,colormap=None):
    # fixme: image load by scipy is RGB, not CV2's channel BGR
    import cv2
    for j,_joint in enumerate(joints):
        _x, _y, _visibility = _joint
        if _visibility == 1.0:
            if colormap:
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


def flip_symmetric_keypoints(keypointsOnImage,matchedParts):
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
