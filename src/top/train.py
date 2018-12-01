#stupid python import stuff to work in terminal, don't know how pycharm does it
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
#########

import argparse
import os
import tensorflow as tf
from keras import backend as k
from net.hourglass import HourglassNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpuID", default=0, type=int, help='gpu id')
    parser.add_argument("--batch_size", default=8, type=int, help='batch size for training')
    parser.add_argument("--model_path",  default="../../trained_models/hg_s2_b1_m",help='path to store trained model')
    parser.add_argument("--data_path", default="../../data", help='path where data is stored')
    parser.add_argument("--num_stack",  default=1, type=int, help='num of stacks')
    parser.add_argument("--epochs", default=1, type=int, help="number of traning epochs")
    parser.add_argument("--data", default=2, type=int, help="data set and processing to use")

    #add aruguemnt for model and data type

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuID)


    # TensorFlow wizardry
    config = tf.ConfigProto()

    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True

    # Only allow a total of half the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = 1.0

    # Create a session with the above options specified.
    k.tensorflow_backend.set_session(tf.Session(config=config))


    # if args.resume:
    #     xnet.resume_train(batch_size=args.batch_size, model_json=args.resume_model_json, model_weights=args.resume_model,
    #                       init_epoch=args.init_epoch, epochs=args.epochs)
    #MPII old, MPII new amd LSP
    data=args.data
    from data_gen.lsp_datgen import LSP_dataset
    from data_gen.mpII_datagen2 import MPII_dataset
    import net.mobilenetv2 as mnet
    if data==0:
        xnet = HourglassNet(num_classes=16, num_hgstacks=args.num_stack, inres=(256, 256), outres=(64, 64))

        xnet.build_model()

        xnet.train_old(epochs=args.epochs, model_path=args.model_path, batch_size=args.batch_size)
    elif data==1:
        xnet = HourglassNet(num_classes=14, num_hgstacks=args.num_stack, inres=(256, 256), outres=(64, 64))
        xnet.build_model()
        xnet.train(MPII_dataset,epochs=args.epochs, model_path=args.model_path, data_path=args.data_path,
                       batch_size=args.batch_size)
    elif data==2:
        xnet = HourglassNet(num_classes=14, num_hgstacks=args.num_stack, inres=(256, 256), outres=(64, 64))
        xnet.build_model()
        xnet.train(LSP_dataset,epochs=args.epochs, model_path=args.model_path,data_path=args.data_path, batch_size=args.batch_size)
    elif data==3:
        net=mnet.MobileNetV2(num_classes=14, inres=(256, 256))
        net.build_model()
        net.train(LSP_dataset, epochs=args.epochs, model_path=args.model_path, data_path=args.data_path,
                   batch_size=args.batch_size)

