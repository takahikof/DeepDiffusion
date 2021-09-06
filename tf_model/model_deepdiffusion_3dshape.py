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

sys.path.append( "tf_model" );
sys.path.append( "utils" );

from ops import *
import retrieval
from transform_3d import *
import arch_3dshape
from feature_augmentation import augment_feature

############################
evaluation_epoch = 10;             # Evaluation on testing dataset is done every 10 epochs
do_save = False;                   # If True, DNN parameters that produces best MAP score is saved to file
do_load = False;                   # If True, the code tries to initialize DNN with pre-trained parameters
############################


class Model_DeepDiffusion_3DShape() :

    ############################
    def __init__( self, args ) :

        self.n_class = args.num_class;
        self.learned_model_dirname = args.save_dir;
        self.device_id = args.device_id;
        self.epoch_training = args.epoch_training;
        self.minibatch_size = args.minibatch_size;
        self.n_dim_embed = args.num_embdim;
        self.n_points_per_shape = args.num_point;
        self.param_knn = args.lmr_knn;
        self.param_lambda = args.lmr_lambda;
        self.encoder_architecture = args.encoder_arch;

        # Load files
        print( "[*] Loading files..." );

        f = h5py.File( args.train_data );
        self.D_train = f['data'][:];
        self.D_train = self.D_train[ :, :, 0:3 ]; # remove normal vectors
        self.L_train = f['label'][:]; # not used
        self.N_train = f['name'][:];
        f.close();

        f = h5py.File( args.test_data );
        self.D_test = f['data'][:];
        self.D_test = self.D_test[ :, :, 0:3 ]; # remove normal vectors
        self.L_test = f['label'][:];
        self.N_test = f['name'][:];
        f.close();

        self.n_train_data = self.D_train.shape[ 0 ];
        self.n_test_data = self.D_test.shape[ 0 ];
        self.n_channel_input = self.D_train.shape[ 2 ];

        # Normalize oriented point sets
        print( "[*] Normalizing data..." );
        for i in range( self.n_train_data ) :
            self.D_train[ i ] = self.normalize_data( self.D_train[ i ] );
        for i in range( self.n_test_data ) :
            self.D_test[ i ] = self.normalize_data( self.D_test[ i ] );

        # sources of diffusion, which represented as one-hot vectors
        self.n_diffusion_source = self.n_train_data;
        self.S_train = np.arange( self.n_diffusion_source );

        print( "[*] Defining network, loss, and optimizer..." );
        self.x_data, self.x_diffusion_source, self.is_training, self.feature_retrieval, self.WeightMat = self.define_network();
        self.loss = self.define_loss();
        self.train_step = self.define_optimizer();

        return;


    ############################
    def define_network( self ) :
        with tf.device( self.device_id ) :
            x_data = tf.placeholder( tf.float32, shape=[ self.minibatch_size, self.n_points_per_shape, self.n_channel_input ] );
            x_diffusion_source = tf.placeholder( tf.float32, shape=[ self.minibatch_size, self.n_diffusion_source ] );
            is_training = tf.placeholder( tf.bool );

            if( self.encoder_architecture == "PointNet" ) :
                y = arch_3dshape.pointnet( x_data, is_training );
            elif( self.encoder_architecture == "DGCNN" ) :
                y = arch_3dshape.dgcnn( x_data, is_training );
            else :
                print( "error: inalid encoder architecture: " + encoder_architecture );
                quit();

            y = linear( y, self.n_dim_embed, "global_fc2" );

            feature_retrieval = l2normalize( y ); # this feature is used for retrieval experiment

            # weight matrix for the final layer
            with tf.variable_scope( "pred" ) :
                WeightMat = tf.get_variable( "matrix", [ self.n_dim_embed, self.n_diffusion_source ], tf.float32,
                                             initializer=tf.contrib.layers.variance_scaling_initializer( factor=2.0, mode='FAN_IN', uniform=False ) );
            WeightMat = tf.transpose( WeightMat );

        return( x_data, x_diffusion_source, is_training, feature_retrieval, WeightMat );


    ############################
    def define_loss( self ) :
        with tf.device( self.device_id ) :

            # define the Latent Manifold Ranking loss

            ### the smoothness term computed by using the extrinsic features (mini-batch latent features) and the intrinsic features (weights in the final layer) ##############
            # find k-nearest neighbors
            S = pairwise_distances( self.feature_retrieval, self.WeightMat, "COSsim" );
            W = ksparse( S, k=self.param_knn );
            mask = tf.not_equal( W, tf.zeros_like( W ) );
            nonzero_idx = tf.where( mask );
            nonzero_val = tf.reshape( tf.boolean_mask( W, mask ), [ -1, 1 ] );

            # generate the ranking vectors
            a = tf.gather( self.feature_retrieval, nonzero_idx[ :, 0 ] );
            b = tf.gather( self.WeightMat, nonzero_idx[ :, 1 ] );
            a = tf.nn.softmax( tf.matmul( a, tf.transpose( self.WeightMat ) ) );
            b = tf.nn.softmax( tf.matmul( b, tf.transpose( self.WeightMat ) ) );

            # compute the Jensen–Shannon divergence
            a = tf.clip_by_value( a, 1e-6, 1.0 );
            b = tf.clip_by_value( b, 1e-6, 1.0 );
            a = tf.distributions.Categorical(probs=a);
            b = tf.distributions.Categorical(probs=b);
            kld_ab = tf.reshape( tf.distributions.kl_divergence( a, b ), [ -1, 1 ] );
            kld_ba = tf.reshape( tf.distributions.kl_divergence( b, a ), [ -1, 1 ] );
            jsd = kld_ab + kld_ba;

            loss_smoothness = tf.reduce_sum( tf.multiply( nonzero_val, jsd ) );
            ################################################################################################

            ### the fitting term computed by using the extrinsic features and their corresponding diffusion source vectors ##########################
            y_predicted = tf.matmul( self.feature_retrieval, tf.transpose( self.WeightMat ) );
            loss_fitting = tf.reduce_sum( tf.nn.softmax_cross_entropy_with_logits( labels=self.x_diffusion_source, logits=y_predicted ) );
            ############################################################

            loss = self.param_lambda * loss_smoothness + loss_fitting;

        return loss;


    ############################
    def define_optimizer( self ) :
        with tf.device( self.device_id ) :
            optimizer = tf.train.AdamOptimizer( 1e-4 );
            train_step = optimizer.minimize( self.loss );
        return train_step;


    ############################
    def normalize_data( self, pointset ) :
        # Normalizes a point set

        # Normalizes position
        mean = np.mean( pointset, axis=0 );
        pointset = pointset - mean;

        # Normalizes scale of a 3D model so that it is enclosed by a sphere whose radius is 0.5
        radius = 0.5;
        norms = np.linalg.norm( pointset, axis=1 );
        max_norm = np.max( norms );
        pointset = pointset * ( radius / max_norm );

        return pointset;


    ############################
    def preprocess_minibatch( self, minibatch, sources, do_data_augmentation ) :

        minibatch_size = minibatch.shape[ 0 ]; # can be less than self.minibatch_size
        n_points = minibatch.shape[ 1 ];
        preprocessed_minibatch = [];

        for i in range( minibatch_size ) :

            # subsample points
            idxs = np.arange( n_points );
            np.random.shuffle( idxs );
            idxs = idxs[ : self.n_points_per_shape ];
            pointset_sub = copy.deepcopy( minibatch[ i ][ idxs ] );

            if( do_data_augmentation ) : # Online data augmentation

                if( np.random.rand() > 0.2 ) :

                    # affine transformation
                    R = generate_randomrot_matrix( angle_max = 5 );
                    Sh = generate_randomshear_matrix( param_max = 0.2 );
                    Sc = generate_randomscale_matrix( param_max = 0.2 );
                    T = generate_randomtranslate_matrix( param_max = 0.2 );
                    transform_matrix = np.matmul( np.matmul( np.matmul( R, Sh ), Sc ), T );
                    pointset_sub = np.concatenate( [ pointset_sub, np.ones( ( self.n_points_per_shape, 1 ) ) ], axis = 1 );
                    pointset_sub = np.matmul( pointset_sub, transform_matrix );
                    pointset_sub = pointset_sub[ :, 0:3 ];

                    if( np.random.rand() > 0.5 ) :
                        # additive noise
                        noise = np.random.normal( 0.0, 0.02, pointset_sub.shape );
                        pointset_sub += noise;

                else :
                    pass;

            preprocessed_minibatch.append( pointset_sub );

        preprocessed_minibatch = np.asarray( preprocessed_minibatch, dtype = np.float32 );

        # create one hot vectors
        if( sources is not None ) :
            preprocessed_sources = np.zeros( ( minibatch_size, self.n_diffusion_source ) );
            rows = np.arange( minibatch_size );
            preprocessed_sources[ rows, sources ] = 1.0;
        else :
            preprocessed_sources = None;

        return( preprocessed_minibatch, preprocessed_sources );


    ############################
    def extract_feature( self, data, feature ) :

        n_data_all = data.shape[ 0 ];
        minibatch_eval = int( np.ceil( float( n_data_all ) / float( self.minibatch_size ) ) );

        F = [];

        for i in range( minibatch_eval ) :
            minibatch = data[ i * self.minibatch_size : (i+1) * self.minibatch_size ];
            minibatch, _ = self.preprocess_minibatch( minibatch, None, False );

            ndata = minibatch.shape[ 0 ];
            if( ndata < self.minibatch_size ) :
                # add dummy data to mini-batch
                dummy_data = np.zeros( [ ( self.minibatch_size - ndata ), self.n_points_per_shape, self.n_channel_input ], dtype="float32" );
                minibatch = np.concatenate( ( minibatch, dummy_data ), axis=0 );

            result = self.sess.run( [ feature ], feed_dict={ self.x_data : minibatch, self.is_training : False } );

            if( ndata < self.minibatch_size ) :
                # remove dummy data
                result[ 0 ] = result[ 0 ][ 0:ndata ];

            F.append( result[ 0 ] );

        F = np.concatenate( F, axis=0 );
        return F;


    ############################
    def evaluate( self, epoch ) :
        weight_train = self.sess.run( self.WeightMat );
        feat_test_embedded = self.extract_feature( self.D_test, self.feature_retrieval );

        retacc = retrieval.retrieval( feat_test_embedded, self.L_test, calc_rpcurve=False );

        map = retacc[ 0 ];
        if( map > self.map_current_best ) :
            best_retacc_is_obtained = True;
            self.best_epoch = epoch;
        else :
            best_retacc_is_obtained = False;

        self.map_current_best = max( [ map, self.map_current_best ] );

        print( "MAP of DD(E) features: " + str( map ) );
        print( "current best MAP was obtained in " + str( self.best_epoch ) + "-th epoch" );
        print( "current best MAP: " + str( self.map_current_best ) );

        if( best_retacc_is_obtained ) :
            print( "computing DD(D+E) features..." )
            feat_test_diffused = augment_feature( weight_train, feat_test_embedded, knn=self.param_knn, knn_for_manifold=self.param_knn );
            retacc = retrieval.retrieval( feat_test_diffused, self.L_test, calc_rpcurve=False );
            map = retacc[ 0 ];
            print( "MAP of DD(D+E) features: " + str( map ) )

        if( best_retacc_is_obtained and do_save ) :
            print(" [*] Saving checkpoint to " + self.learned_model_dirname );
            if not os.path.exists( self.learned_model_dirname ) :
                os.makedirs( self.learned_model_dirname );
            step = 0; # in order to overwrite existing file
            saver.save( self.sess, os.path.join( self.learned_model_dirname, "dnn" ), global_step=step );


    ############################
    def train_and_test( self ) :

        print( "[*] Initizalizing network parameters..." );
        config = tf.ConfigProto( allow_soft_placement = True, log_device_placement = False );
        config.gpu_options.allow_growth = True;
        self.sess = tf.Session( config = config );
        saver = tf.train.Saver();

        # initialize parameters of DNN
        self.sess.run( tf.initialize_all_variables() );

        # initialize parameters of the final layer with features extracted from training data
        print( "[*] Initializing the intrinsic features (i.e., parameters of the final layer) with features extracted from training samples..." );
        feat_train = self.extract_feature( self.D_train, self.feature_retrieval );
        init_weight = tf.placeholder( tf.float32, shape=[ feat_train.shape[1], feat_train.shape[0] ] );
        init_target = [ v for v in tf.trainable_variables() if "pred/matrix" in v.name ][ 0 ];
        assign_op = tf.assign( init_target, init_weight );
        self.sess.run( assign_op, feed_dict={ init_weight : np.transpose( feat_train ) } );

        if( do_load ) :
            print(" [*] Reading checkpoint from " + self.learned_model_dirname );
            ckpt = tf.train.get_checkpoint_state( self.learned_model_dirname );
            if ckpt and ckpt.model_checkpoint_path :
                ckpt_name = os.path.basename( ckpt.model_checkpoint_path );
                saver.restore( self.sess, os.path.join( self.learned_model_dirname, ckpt_name ) );
                print( "Pretrained parameters exist. Start training with pretrained parameters." );
            else:
                print( "Pretrained parameters do not exist. Start training with randomly initialized parameters." );


        print( "[*] Start training." );

        minibatch_n_train = int( self.n_train_data / self.minibatch_size );    # Number of training mini-batches per epoch
        if( minibatch_n_train == 0 ) :
            minibatch_n_train = 1;

        print( "Number of training data : " + str( self.n_train_data ) );
        print( "Number of testing data : " + str( self.n_test_data ) );
        print( "Number of training minibatches per epoch : " + str( minibatch_n_train ) );

        print( "epoch_training : " + str( self.epoch_training ) );
        print( "minibatch_size : " + str( self.minibatch_size ) );

        self.best_epoch = 0;
        self.map_current_best = 0.0;

        for i in range( self.epoch_training ) :
            print( "epoch : " + str( i ) );

            # shuffle training data
            modelID_list = list( range( self.n_train_data ) );
            np.random.shuffle( modelID_list );
            D_train_shuffled = self.D_train[ modelID_list ];
            L_train_shuffled = self.L_train[ modelID_list ];
            N_train_shuffled = self.N_train[ modelID_list ];
            S_train_shuffled = self.S_train[ modelID_list ];

            if( i >= 0 ) : # compute current loss by using subset of training data
                loss_total = 0.0;
                n_processed = 0;
                minibatch_n_train_sub = int( minibatch_n_train / 10.0 ) if int( minibatch_n_train / 10.0 ) > 0 else 1; # only 1/10 data are used
                for j in range( minibatch_n_train_sub ) :
                    minibatch_data = D_train_shuffled[ j * self.minibatch_size : (j+1) * self.minibatch_size ];
                    minibatch_source = S_train_shuffled[ j * self.minibatch_size : (j+1) * self.minibatch_size ];
                    minibatch_data, minibatch_source = self.preprocess_minibatch( minibatch_data, minibatch_source, False );

                    result = self.sess.run( [ self.loss ], feed_dict={ self.x_data : minibatch_data,
                                                                  self.x_diffusion_source : minibatch_source,
                                                                  self.is_training : False } );
                    loss_total += result[ 0 ];
                    n_processed += self.minibatch_size;

                print( "[training data] loss : " + str( loss_total / n_processed ) );

            if( i % evaluation_epoch == 0 ) : # evaluate current latent feature
                print( "[*] Evaluating retrieval accuracy using the testing dataset." );
                self.evaluate( i );

            # train
            for j in range( minibatch_n_train ) :
                minibatch_data = D_train_shuffled[ j * self.minibatch_size : (j+1) * self.minibatch_size ];
                minibatch_source = S_train_shuffled[ j * self.minibatch_size : (j+1) * self.minibatch_size ];
                minibatch_data, minibatch_source = self.preprocess_minibatch( minibatch_data, minibatch_source, True );

                self.sess.run( self.train_step, feed_dict={ self.x_data : minibatch_data,
                                                            self.x_diffusion_source : minibatch_source,
                                                            self.is_training : True } );

        print( "[*] Evaluating retrieval accuracy using the testing dataset." );
        self.evaluate( self.epoch_training );

        return;
