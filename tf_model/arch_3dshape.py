# -*- coding: utf-8 -*-
import tensorflow as tf
import sys
sys.path.append( "tf_model" );
from ops import *
import dgcnn_util

def dgcnn( x_data, is_training, bn=True ) :
  # borrowed from: https://github.com/WangYueFt/dgcnn/blob/master/tensorflow/models/dgcnn.py
  batch_size = x_data.get_shape()[0].value
  num_point = x_data.get_shape()[1].value
  k = 20
  bn_decay = None

  adj_matrix = dgcnn_util.pairwise_distance( x_data )
  nn_idx = dgcnn_util.knn(adj_matrix, k=k)
  edge_feature = dgcnn_util.get_edge_feature( x_data, nn_idx=nn_idx, k=k)

  net = dgcnn_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=bn, is_training=is_training,
                       scope='dgcnn1', bn_decay=bn_decay)
  net = tf.reduce_max(net, axis=-2, keep_dims=True)
  net1 = net

  adj_matrix = dgcnn_util.pairwise_distance(net)
  nn_idx = dgcnn_util.knn(adj_matrix, k=k)
  edge_feature = dgcnn_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

  net = dgcnn_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=bn, is_training=is_training,
                       scope='dgcnn2', bn_decay=bn_decay)
  net = tf.reduce_max(net, axis=-2, keep_dims=True)
  net2 = net

  adj_matrix = dgcnn_util.pairwise_distance(net)
  nn_idx = dgcnn_util.knn(adj_matrix, k=k)
  edge_feature = dgcnn_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

  net = dgcnn_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=bn, is_training=is_training,
                       scope='dgcnn3', bn_decay=bn_decay)
  net = tf.reduce_max(net, axis=-2, keep_dims=True)
  net3 = net

  adj_matrix = dgcnn_util.pairwise_distance(net)
  nn_idx = dgcnn_util.knn(adj_matrix, k=k)
  edge_feature = dgcnn_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

  net = dgcnn_util.conv2d(edge_feature, 128, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=bn, is_training=is_training,
                       scope='dgcnn4', bn_decay=bn_decay)
  net = tf.reduce_max(net, axis=-2, keep_dims=True)
  net4 = net

  net = dgcnn_util.conv2d(tf.concat([net1, net2, net3, net4], axis=-1), 1024, [1, 1],
                       padding='VALID', stride=[1,1],
                       bn=bn, is_training=is_training,
                       scope='agg', bn_decay=bn_decay)

  net = tf.reduce_max(net, axis=1, keep_dims=True)

  # MLP on global point cloud vector
  net = tf.reshape(net, [batch_size, -1])
  net = dgcnn_util.fully_connected(net, 512, bn=bn, is_training=is_training,
                                scope='fc1', bn_decay=bn_decay)
  return net;


def pointnet( x_data, is_training, bn=True ) :

    minibatch_size = x_data.get_shape().as_list()[0];
    n_points_per_shape = x_data.get_shape().as_list()[1];
    n_channel_input = x_data.get_shape().as_list()[2];

    y = tf.reshape( x_data, [ minibatch_size * n_points_per_shape, 1, 1, n_channel_input ] );

    # per-point feature extraction
    y = conv2d( y, 64, [ 1, 1 ], "perpoint_fc11" );
    if( bn ) :
        y = relu( batch_norm_for_conv2d( y, is_training, None, "bn11" ) );
    else :
        y = relu( y );

    y = conv2d( y, 64, [ 1, 1 ], "perpoint_fc12" );
    if( bn ) :
        y = relu( batch_norm_for_conv2d( y, is_training, None, "bn12" ) );
    else :
        y = relu( y );

    y = conv2d( y, 64, [ 1, 1 ], "perpoint_fc13" );
    if( bn ) :
        y = relu( batch_norm_for_conv2d( y, is_training, None, "bn13" ) );
    else :
        y = relu( y );

    y = conv2d( y, 128, [ 1, 1 ], "perpoint_fc14" );
    if( bn ) :
        y = relu( batch_norm_for_conv2d( y, is_training, None, "bn14" ) );
    else :
        y = relu( y );

    y = conv2d( y, 1024, [ 1, 1 ], "perpoint_fc15" );
    if( bn ) :
        y = relu( batch_norm_for_conv2d( y, is_training, None, "bn15" ) );
    else :
        y = relu( y );

    # aggregation by global max pooling
    y = tf.reshape( y, [ minibatch_size * n_points_per_shape, -1 ] );
    y = aggregate( y, "M", minibatch_size, n_points_per_shape );
    # The above two lines can be replaced with the following two lines.
    # y = tf.reshape( y, [ minibatch_size, n_points_per_shape, -1 ] );
    # y = tf.reduce_max( y, axis=1 );

    # embedding by a fully-connected layer
    y = linear( y, 512, "global_fc1" );
    if( bn ) :
        y = relu( batch_norm_for_fc( y, is_training, None, "bn16" ) );
    else :
        y = relu( y );

    return y;
