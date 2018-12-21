from scipy.io import loadmat
import glob
import re
import os.path
from ntpath import basename
from data_gen.data_gen_utils import *



class LSP_dataset(DataGen):
    image_dir, joint_file = "lspet/images", "lspet/joints.mat"
    transpose_order = (2, 0, 1)
    def _load_image_joints(self):
        """
        The file joints.mat is a MATLAB data file containing the joint annotations
          in a 3x14x10000 matrix called 'joints' with x and y locations
          and a binary value indicating the visbility of each joint

        Create a list of lines in format:
          image_path, x1, y1, x2,y2, ...
          where xi, yi - coordinates of the i-th joint
        """
        joints = loadmat(self.joint_file)
        joints = joints['joints'].transpose(*self.transpose_order)
        lines = list()
        out = np.empty(len(joints), dtype=object)

        for img_path in sorted(glob.glob(os.path.join(self.image_dir, '*.jpg'))):
            index = int(re.search(r'im([0-9]+)', basename(img_path)).groups()[0]) - 1
            img_joint = [basename(img_path), joints[index]]
            lines.append(img_joint)
        out[:] = lines
        return out

if __name__ == "__main__":
    image_dir, joint_file = "../../data/lspet/images", "../../data/lspet/joints.mat"
    data_set = LSP_dataset(image_dir, joint_file, [128,128], [128,128], 1)
    data_set.test_visualise()


