# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import sys
import time
import random
import math
import os
import h5py
import copy
import warnings
import argparse
from distutils.util import strtobool

sys.path.append( "tf_model" );
import model_deepdiffusion_3dshape as model

###########################
if( __name__ == '__main__') :

    parser = argparse.ArgumentParser();
    parser.add_argument('--device_id', type=str, default="/gpu:0", help='device (e.g., /cpu:0, /gpu:0, /gpu:1, ...)');
    parser.add_argument('--train_data', type=str, default='', help='path to the training dataset');
    parser.add_argument('--test_data', type=str, default='', help='path to the testing dataset');
    parser.add_argument('--epoch_training', type=int, default=300, help='number of epochs for training');
    parser.add_argument('--minibatch_size', type=int, default=32, help='minibatch size');
    parser.add_argument('--save_dir', type=str, default='', help='path to the directory where a learned DNN is saved');
    parser.add_argument('--num_class', type=int, default=40, help='number of classes');
    parser.add_argument('--num_embdim', type=int, default=256, help='number of dimensions for the latent feature space');
    parser.add_argument('--encoder_arch', type=str, default='PointNet', help='encoder architecture (PointNet or DGCNN)');
    parser.add_argument('--lmr_knn', type=int, default=20, help='number of nearest neighbors for the LMR loss');
    parser.add_argument('--lmr_lambda', type=float, default=1.0, help='coefficient for the LMR loss');
    parser.add_argument('--num_point', type=int, default=1024, help='number of points per 3D shape');

    args = parser.parse_args();

    print( args );

    np.set_printoptions( threshold = np.inf, precision=4 );
    if not sys.warnoptions : # disable warnings
        warnings.simplefilter("ignore");
        os.environ["PYTHONWARNINGS"] = "ignore";

    MODEL = model.Model_DeepDiffusion_3DShape( args );

    MODEL.train_and_test();

    quit();
