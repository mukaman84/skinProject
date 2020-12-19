import os
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
# from tensorflow as keras
import tensorflow as tf
import numpy as np

from dataloader import *
import albumentations as A
import segmentation_models as sm




def inference(args):

    BACKBONE = 'resnet101'
    BATCH_SIZE = 4
    CLASSES = ['acne']
    LR = 0.0001
    EPOCHS = 1000

    sm.set_framework('tf.keras')
    preprocess_input = sm.get_preprocessing(BACKBONE)

    # define network parameters
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
    activation = 'sigmoid' if n_classes == 1 else 'softmax'

    # create model
    model = sm.FPN(BACKBONE, classes=n_classes, activation=activation)
    model.load_weights('./best_model_{:s}.h5'.format(args.target))
    print('./best_model_{:s}.h5'.format(args.target))

    if os.path.isdir(args.testPath):
        if not os.path.isdir(args.outputPath):
            os.mkdir(args.outputPath,0o777)
            os.chmod(args.outputPath, 0o777)


        listSet = os.listdir(args.testPath)
        for test in listSet:
            if test.endswith('.png') or test.endswith('.jpg'):
                image = cv2.imread(os.path.join(args.testPath,test))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = np.expand_dims(image, axis=0)
                pr_mask = model.predict(image/255).round()*255
                pr_mask = np.concatenate((np.zeros_like(pr_mask[0,...]),np.zeros_like(pr_mask[0,...]),pr_mask[0,...]),axis=2)
                output_path = args.outputPath + test
                cv2.imwrite(output_path,pr_mask)
                os.chmod(output_path,0o777)
    else:
        image = cv2.imread(os.path.join(args.testPath))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('111',image)
        # cv2.waitKey(0)
        # image = image / 255
        image = np.expand_dims(image, axis=0)
        pr_mask = model.predict(image).round()*255
        # pr_mask = model.predict(image)
        a= np.max(pr_mask)
        pr_mask = np.concatenate((np.zeros_like(pr_mask[0, ...]), np.zeros_like(pr_mask[0, ...]), pr_mask[0, ...]),
                                 axis=2)
        output_path = args.outputPath + os.path.basename(args.testPath)
        cv2.imwrite(output_path, pr_mask)
        os.chmod(output_path, 0o777)

    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='inference', type=str,
                        help="train or test?")
    parser.add_argument('--testPath', default='dataset/acne/JPEGImages/temp_.jpeg', type=str,
                        help="path to test set")
    parser.add_argument('--outputPath', default='data/acne/output/', type=str,
                        help="path to test set")
    parser.add_argument('--target', default='acne', type=str,
                        help="target name?: ex) acne, hemo, mela")
    args = parser.parse_args()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    tf.keras.backend.set_session(session)
    inference(args)
    # if args.mode == 'inference':



if __name__=="__main__":
    main()
