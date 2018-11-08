import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from net.hourglass import HourglassNet

from eval.heatmap_process import post_process_heatmap
import argparse

from data_gen.mpii_datagen import MPIIDataGen
import cv2

import os
cwd = os.getcwd()
print(cwd)
def render_joints(cvmat, joints, conf_th=0.2):
    for _joint in joints:
        _x, _y , _conf = _joint
        if _conf > conf_th:
            cv2.circle(cvmat, center=(int(_x), int(_y)), color=(255, 0, 0), radius=7, thickness=2)

    return cvmat

def main_inference(model_json, model_weights, num_stack, num_class, imgfile, confth):
    xnet = HourglassNet(num_class, num_stack, (256, 256), (64, 64))
    xnet.load_model(model_json, model_weights)

    out, scale = xnet.inference_file(imgfile)

    kps = post_process_heatmap(out[0,:,:,:])

    mkps = list()
    for i, _kp in enumerate(kps):
        _conf = _kp[2]
        mkps.append((_kp[0]*scale[1]*4, _kp[1]*scale[0]*4, _conf))

    cvmat = render_joints(cv2.imread(imgfile), mkps, confth)

    cv2.imshow('frame', cvmat)
    cv2.waitKey()


def main_video(model_json, model_weights, num_stack, num_class, videofile, confth):

    xnet = HourglassNet(num_class, num_stack, (256, 256), (64, 64))
    xnet.load_model(model_json, model_weights)

    cap = cv2.VideoCapture(videofile)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            rgb = frame[:,:,::-1] # bgr -> rgb
            out, scale = xnet.inference_rgb(rgb, frame.shape)

            kps = post_process_heatmap(out[0, :, :, :])

            ignore_kps = ['plevis', 'thorax', 'head_top']
            kp_keys = MPIIDataGen.get_kp_keys()
            mkps = list()
            for i, _kp in enumerate(kps):
                if kp_keys[i] in ignore_kps:
                    _conf = 0.0
                else:
                    _conf = _kp[2]
                mkps.append((_kp[0] * scale[1] * 4, _kp[1] * scale[0] * 4, _conf))

            framejoints = render_joints(frame, mkps, confth)

            cv2.imshow('frame', framejoints)
            cv2.waitKey(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpuID", default=0, type=int, help='gpu id')
    parser.add_argument("--model_json", default="../../trained_models/hg_s2_b1_m/net_arch.json", help='path to store trained model')
    parser.add_argument("--model_weights", default="../../trained_models/hg_s2_b1_m/weights_epoch0.h5", help='path to store trained model')
    parser.add_argument("--num_stack",  type=int, help='num of stack')
    parser.add_argument("--input_image",  default="../../images/sample.jpg",help='input image file')
    parser.add_argument("--input_video", default='', help='input video file')
    parser.add_argument("--n_joints", type=int, default=14, help='number of joints')
    parser.add_argument("--conf_threshold", type=float, default=0.2, help='confidence threshold')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuID)

    if args.input_image:
       main_inference(model_json=args.model_json, model_weights=args.model_weights, num_stack=args.num_stack,
                   num_class=args.n_joints, imgfile = args.input_image, confth=args.conf_threshold)
    elif args.input_video:
        main_video(model_json=args.model_json, model_weights=args.model_weights, num_stack=args.num_stack,
                   num_class=args.n_joinst, videofile=args.input_video, confth=args.conf_threshold)