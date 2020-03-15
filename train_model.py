import sys
import tensorflow as tf
import tensorflow.keras
from datetime import datetime
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, History
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Activation
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import multi_gpu_model
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
import argparse

SCRIPT_PATH = os.path.dirname(__file__)  # script directory

def data_loader(path,input_shape):
    data_frame = pd.read_csv(path)
    # Extract images
    data_frame['image'] = data_frame['image'].apply(
        lambda i: np.fromstring(i, sep=' '))
    data_frame = data_frame.dropna()  # Get only the data with 15 keypoints
    imgs_array = np.vstack(data_frame['image'].values)/255.0
    # Normalize, target values to (0, 1)
    imgs_array = imgs_array.astype(np.float32)
    imgs_array = imgs_array.reshape(-1,input_shape, input_shape, 1)
    # Extract labels (key point cords)
    labels_array = data_frame[data_frame.columns[:-1]].values
    labels_array = (labels_array - input_shape/2) / input_shape/2    # Normalize, traget cordinates to (-1, 1)
    labels_array = (labels_array)     # Normalize, traget cordinates to (-1, 1)
    labels_array = labels_array.astype(np.float32)
#     shuffle the train data
    imgs_array, labels_array = shuffle(
        imgs_array, labels_array, random_state=9)
    return imgs_array, labels_array


def facial_landmark_model():
    model = Sequential()
    # Layer 1
    model.add(Conv2D(32, (3, 3), padding='same',
                     activation='relu', input_shape=X_train.shape[1:]))
    model.add(MaxPooling2D(pool_size=2))  # 64*64

    # Layer 2
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))  # 32*32

    # Layer 3
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))  # 16*16

    # Layer 4
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))  # 8*8

    # Layer 5
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))  # 4*4

    # Layer 6
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))  # 2*2

    # Layer 7
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
#    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(196))  # 98 facial landmarks points

    return model

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True,
                    help="Path to dataset (98 facial landmarks dataset is compatible for this file)", type=str)
    ap.add_argument("-e", "--epoch", required=True,
                    help="number of epoch to train a model", type=int)
    ap.add_argument("-lr", "--learning-rate", required=False,
                    default=0.001, type=float, help="Learning rate for adam optimizer")

    ap.add_argument("-n", "--num-gpu", required=False,
                    default=1, type=int, help="Number of gpu to use for training")
    ap.add_argument("-c", "--checkpoint-path", required=True,
                    help="Path to checkpoint file", type=str)
    ap.add_argument("-m", "--model-path", required=True,
                    help="Path to model file", type=str)
    ap.add_argument("-i", "--iteration", required=False, default=100,
                    help="Path to model file", type=int)
    ap.add_argument("--metrics",type=str,required=True,help="Set metrics for the optimizer")
    ap.add_argument("--monitor",type=str,required=True,help="Set vairable to be monitored")
    ap.add_argument("--input-shape",required=False,type=int,help="shape of training set input image")
    ap.add_argument("--gpu",type=str,required=False,default="0",help="GPU used to train model")
    ap.add_argument("-vs","--validation-split",default=0.3,type=float,required=False,help="Validation split for training set")
    ap.add_argument("-bs","--batch-size",default=64,type=int,required=False,help="Configure batch size for each training iteration")
    args = vars(ap.parse_args())
    BATCH_SIZE = args["batch_size"] # Configure batch size for each training iteration
    VALIDATION_SPLIT = args["validation_split"] # portion of test set and training set
    GPU_INDEX = args["gpu"] # GPU used for training model
    INPUT_SHAPE = args["input_shape"] # input shape of training set
    METRICS = args["metrics"].split(",") # metrics to be used for optimizer
    MONITOR = args["monitor"] # value to be monitored and captured
    TRAINING_ITERATION = args["iteration"]  # iteration to train model
    PATH_TO_DATASET = args["path"]  # path to dataset
    EPOCHS = args["epoch"]  # number of epoch to train in each iteration
    LEARNING_RATE = args["learning_rate"]  # learning rate of model
    NUM_GPU = args["num_gpu"]  # number of gpu used to train model
    CHECKPOINT_PATH = args["checkpoint_path"]  # path to checkpoint
    CHECKPOINT_PATH = os.path.join(
        SCRIPT_PATH, CHECKPOINT_PATH)  # path to checkpoint file
    MODEL_PATH = args["model_path"]  # path to model
    MODEL_PATH = os.path.join(
        SCRIPT_PATH, MODEL_PATH)  # path to model file
    MODEL_EXIST = os.path.exists(MODEL_PATH)  # check if model file exist

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=GPU_INDEX
    print ("TRAINING CONFIGURATION")
    print(args)
    print("-"*100)
    # input validation
    if NUM_GPU < 1:
        NUM_GPU = 1

    print("Loading Dataset For Training...")
    X_train, y_train = data_loader(PATH_TO_DATASET,INPUT_SHAPE)
    print("Training datapoint shape: X_train.shape:{}".format(X_train.shape))
    print("Training labels shape: y_train.shape:{}".format(y_train.shape))
    model = facial_landmark_model()
    if MODEL_EXIST:
        print("Model Found...")
        print("Continue Training From Existed Model ...")
        model = load_model(MODEL_PATH)
    hist = History()
    checkpointer = ModelCheckpoint(filepath=CHECKPOINT_PATH,monitor=MONITOR,
                                   verbose=1, save_best_only=True)
    # Complie Model
    epochs = EPOCHS  # Epcoh of training to go
    adam_optimizer = optimizers.Adam(lr=LEARNING_RATE)  # ADam optimizer
    # use single gpu to train model
    if NUM_GPU == 1:
        for i in range(TRAINING_ITERATION):
            model.compile(optimizer=adam_optimizer, loss='mean_squared_error',
                          metrics=METRICS)
            model_fit = model.fit(X_train, y_train, validation_split=VALIDATION_SPLIT, epochs=epochs, shuffle=True,
                                  batch_size=BATCH_SIZE, callbacks=[checkpointer, hist], verbose=1)
            model.save(MODEL_PATH)
            print("SUCCESSFULLY SAVED MODEL --- ITERATION {}".format(i))
    # if using multiple gpu to train model
    elif NUM_GPU >= 2:
        parallel_model = multi_gpu_model(
            model, gpus=NUM_GPU, cpu_merge=True)
        parallel_model.compile(loss='mean_squared_error',
                               optimizer=adam_optimizer, metrics=['accuracy'])
        parallel_model.fit(X_train, y_train, validation_split=0.30, epochs=epochs,
                           batch_size=batch_size, callbacks=[checkpointer, hist], verbose=1)
        parallel_model.save('models/model_128_9.h5')
        print("Save model successfully")

