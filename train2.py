import os
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import cv2
# from tensorflow as keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dataloader import *
import albumentations as A
import segmentation_models as sm


def inference(args):
    return

def train(target):
    DATA_DIR = '/root/skinProject/data/' + target + '/'

    x_train_dir = os.path.join(DATA_DIR, 'JPEGImages')
    y_train_dir = os.path.join(DATA_DIR, 'SegmentationClassPNG')

    x_valid_dir = os.path.join(DATA_DIR, 'JPEGImages')
    y_valid_dir = os.path.join(DATA_DIR, 'SegmentationClassPNG')

    x_test_dir = os.path.join(DATA_DIR, 'JPEGImages')
    y_test_dir = os.path.join(DATA_DIR, 'SegmentationClassPNG')

    # dataset = Dataset(x_train_dir, y_train_dir, classes=['car', 'pedestrian'])

    # Lets look at augmented data we have
    dataset = Dataset(x_train_dir, y_train_dir, classes=[target], augmentation=get_training_augmentation())

    # image, mask = dataset[12] # get some sample
    # visualize(
    #     image=image,
    #     cars_mask=mask[..., 0].squeeze(),
    #     sky_mask=mask[..., 1].squeeze(),
    #     background_mask=mask[..., 2].squeeze(),
    # )

    BACKBONE = 'resnet101'
    BATCH_SIZE = 4
    CLASSES = ['acne']
    LR = 0.0001
    EPOCHS = 300

    sm.set_framework('tf.keras')
    preprocess_input = sm.get_preprocessing(BACKBONE)

    # define network parameters
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
    activation = 'sigmoid' if n_classes == 1 else 'softmax'

    # create model
    model = sm.FPN(BACKBONE, classes=n_classes, activation=activation)

    # define optomizer
    optim = tf.keras.optimizers.Adam(LR)

    # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
    # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss

    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    # compile keras model with defined optimozer, loss and metrics
    model.compile(optim, total_loss, metrics)

    # Dataset for train images
    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        classes=CLASSES,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )

    # Dataset for validation images
    valid_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        classes=CLASSES,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )

    train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

    # # check shapes for errors
    # assert train_dataloader[0][0].shape == (BATCH_SIZE, 320, 320, 3)
    # assert train_dataloader[0][1].shape == (BATCH_SIZE, 320, 320, n_classes)

    # define callbacks for learning rate scheduling and best checkpoints saving
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('./best_model_{:s}.h5'.format(target), save_weights_only=True, save_best_only=True, mode='min'),
        tf.keras.callbacks.ReduceLROnPlateau(),
    ]

    # train model
    history = model.fit_generator(
        train_dataloader,
        steps_per_epoch=len(train_dataloader),
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=valid_dataloader,
        validation_steps=len(valid_dataloader),
    )
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str,
                        help="train or test?")
    parser.add_argument('--target', default='acne', type=str,
                        help="target name?: ex) acne, hemo, mela")
    args = parser.parse_args()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    tf.keras.backend.set_session(session)

    if args.mode == 'train':
        TARGETS = ['wrinkle']#['acne', 'hemo', 'mela']
        for tar in TARGETS:
            train(tar)
    else:
        inference(args)


if __name__=="__main__":
    main()