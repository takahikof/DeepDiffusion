# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import math

#################################
def sqrt( x ) : # sqrt(0)の微分値がinfになるため,sqrt(0)を回避
    return tf.sqrt( tf.clip_by_value( x, 1e-8, 1e8 ) );

def tanh( x ) :
    return tf.nn.tanh( x );

def relu( x ) :
    return tf.nn.relu( x );

def dropout( x, keep_prob ) :
    return tf.nn.dropout( x, keep_prob );

def l2normalize( x ) :
    y = tf.nn.l2_normalize( x + 1e-8, 1 );
    return y;

def ksparse( x, k ) :
    values, indices = tf.nn.top_k( x, k = k );
    values = values[ :, k - 1 ];
    values = tf.reshape( values, [ -1, 1 ] );
    y = tf.where( tf.less( x , values ), tf.zeros_like( x ), x );
    return y;

def linear( x, output_size, scope ) :
    shape = x.get_shape().as_list();
    with tf.variable_scope( scope ) :
        matrix = tf.get_variable( "matrix", [shape[1], output_size], tf.float32,
                                  initializer=tf.contrib.layers.variance_scaling_initializer( factor=2.0, mode='FAN_IN', uniform=False ) );
        bias = tf.get_variable( "bias", [output_size], initializer=tf.constant_initializer( 0.0  ))
        return( tf.matmul( x, matrix ) + bias );

def linear_no_bias( x, output_size, scope ) :
    shape = x.get_shape().as_list();
    with tf.variable_scope( scope ) :
        matrix = tf.get_variable( "matrix", [shape[1], output_size], tf.float32,
                                  initializer=tf.contrib.layers.variance_scaling_initializer( factor=2.0, mode='FAN_IN', uniform=False ) );
        return( tf.matmul( x, matrix ) );

def conv2d( x, output_dim, kernel_size, scope, stride=1 ) :
    with tf.variable_scope( scope ) :
        k_h, k_w = kernel_size;
        w = tf.get_variable( "conv", [ k_h, k_w, x.get_shape()[-1], output_dim ],
                            initializer=tf.contrib.layers.variance_scaling_initializer( factor=2.0, mode='FAN_IN', uniform=False ) );
        conv = tf.nn.conv2d( x, w, strides=[1, stride, stride, 1], padding='SAME');
        biases = tf.get_variable( 'biases', [ output_dim ], initializer=tf.constant_initializer(0.0) );
        shape = conv.get_shape().as_list();
        conv = tf.reshape( tf.nn.bias_add( conv, biases ), [ -1, shape[1], shape[2], shape[3] ]  );
        return conv;

def deconv2d( x, output_shape, kernel_size, stride, scope ) :
    with tf.variable_scope( scope ) :
        w = tf.get_variable( 'deconv', [ kernel_size, kernel_size, output_shape[-1], x.get_shape()[-1] ],
                            initializer=tf.contrib.layers.variance_scaling_initializer( factor=2.0, mode='FAN_IN', uniform=False ) );
        deconv = tf.nn.conv2d_transpose( x, w, output_shape=output_shape, strides=[1, stride, stride, 1] );
        biases = tf.get_variable( 'biases', [ output_shape[-1] ], initializer=tf.constant_initializer(0.0) );
        deconv = tf.reshape(tf.nn.bias_add( deconv, biases ), deconv.get_shape() );
        return deconv;

def aggregate( x, mode, minibatch_size, n_lf_permodel ) :
    # xは[ minibatch_size * n_lf_permodel, n_dim_in ]の2D tensor
    shape = x.get_shape().as_list();
    n_dim_in = shape[ 1 ];

    segment_ids = tf.range( minibatch_size );
    segment_ids = tf.reshape( segment_ids, [ -1, 1 ] );
    segment_ids = tf.tile( segment_ids, [ 1, n_lf_permodel ] );
    segment_ids = tf.squeeze( tf.reshape( segment_ids, [ -1, 1 ] ) );

    if( mode == "A" ) : # average pooling
        y = tf.segment_mean( x, segment_ids );
        y = tf.reshape( y, [ minibatch_size, n_dim_in ] ); # segment_mean/segment_maxするとtensorのshapeが不定になるので明らかにしておく
    elif( mode == "M" ) : # max pooling
        y = tf.segment_max( x, segment_ids );
        y = tf.reshape( y, [ minibatch_size, n_dim_in ] ); # segment_mean/segment_maxするとtensorのshapeが不定になるので明らかにしておく

    return y;

def pairwise_distances( A, B, metric ) :

    if( metric == "L2" ) : # L2 distance
        sqnormA = tf.reduce_sum( tf.multiply( A, A ), 1, keep_dims=True );
        sqnormB = tf.reduce_sum( tf.multiply( B, B ), 1, keep_dims=True );
        distmat = sqnormA - 2.0 * tf.matmul( A, tf.transpose( B ) ) + tf.transpose( sqnormB ); # squared euclidean distances
        distmat = sqrt( distmat ); # euclidean distances
    elif( metric == "L1" ) : # L1 distance
        expanded_a = tf.expand_dims( A, 1 );
        expanded_b = tf.expand_dims( B, 0 );
        distmat = tf.reduce_sum( tf.abs( expanded_a - expanded_b ), 2 );
    elif( metric == "L05" ) : # L0.5 distance
        expanded_a = tf.expand_dims( A, 1 );
        expanded_b = tf.expand_dims( B, 0 );
        distmat = tf.square( tf.reduce_sum( sqrt( tf.abs( expanded_a - expanded_b ) ), 2 ) );
    elif( metric == "COS" ) : # Cosine distance
        normA = sqrt( tf.reduce_sum( tf.multiply( A, A ), 1, keep_dims=True ) );
        normB = sqrt( tf.reduce_sum( tf.multiply( B, B ), 1, keep_dims=True ) );
        norms = normA * tf.transpose( normB );
        distmat = tf.matmul( A, tf.transpose( B ) ) / norms;
        distmat = 1.0 - ( 1.0 + distmat ) / 2.0;
    elif( metric == "COSsim" ) : # Cosine similarity
        normA = sqrt( tf.reduce_sum( tf.multiply( A, A ), 1, keep_dims=True ) );
        normB = sqrt( tf.reduce_sum( tf.multiply( B, B ), 1, keep_dims=True ) );
        norms = normA * tf.transpose( normB );
        distmat = tf.matmul( A, tf.transpose( B ) ) / norms;
        distmat = ( 1.0 + distmat ) / 2.0;
    elif( metric == "AD" ) : # Absolute dot
        dots = tf.matmul( A, tf.transpose( B ) );
        distmat = tf.abs( dots );
    elif( metric == "CHI" ) : # Chi-squared distance
        expanded_a = tf.expand_dims( A, 1 );
        expanded_b = tf.expand_dims( B, 0 );
        distmat = sqrt( tf.reduce_sum( tf.square( expanded_a - expanded_b ) / ( expanded_a + expanded_b + 1e-8 ), 2 ) );
    elif( metric == "CAM" ) : # Camberra distance
        expanded_a = tf.expand_dims( A, 1 );
        expanded_b = tf.expand_dims( B, 0 );
        distmat = tf.reduce_sum( tf.abs( expanded_a - expanded_b ) / ( expanded_a + expanded_b + 1e-8 ), 2 );
    elif( metric == "KLD" ) : # symmetric Kullback-Leibler divergence
        pass;
    else :
        print( "invalid metric : " + metric );
        quit();

    return distmat;

def batch_norm_for_fc(inputs, is_training, bn_decay, scope):
  """ Batch normalization on FC data.

  Args:
      inputs:      Tensor, 2D BxC input
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,], bn_decay)

def batch_norm_for_conv1d(inputs, is_training, bn_decay, scope):
  """ Batch normalization on 1D convolutional maps.

  Args:
      inputs:      Tensor, 3D BLC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,1], bn_decay)

def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope):
  """ Batch normalization on 2D convolutional maps.

  Args:
      inputs:      Tensor, 4D BHWC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,1,2], bn_decay)

def batch_norm_for_conv3d(inputs, is_training, bn_decay, scope):
  """ Batch normalization on 3D convolutional maps.

  Args:
      inputs:      Tensor, 5D BDHWC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,1,2,3], bn_decay)

def batch_norm_template(inputs, is_training, scope, moments_dims, bn_decay):
  """ Batch normalization on convolutional maps and beyond...
  Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow

  Args:
      inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
      is_training:   boolean tf.Varialbe, true indicates training phase
      scope:         string, variable scope
      moments_dims:  a list of ints, indicating dimensions for moments calculation
      bn_decay:      float or float tensor variable, controling moving average weight
  Return:
      normed:        batch-normalized maps
  """
  with tf.variable_scope(scope) as sc:
    num_channels = inputs.get_shape()[-1].value
    beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
                       name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
                        name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
    decay = bn_decay if bn_decay is not None else 0.9
    ema = tf.train.ExponentialMovingAverage(decay=decay)
    # Operator that maintains moving averages of variables.
    ema_apply_op = tf.cond(is_training,
                           lambda: ema.apply([batch_mean, batch_var]),
                           lambda: tf.no_op())

    # Update moving average and return current batch's avg and var.
    def mean_var_with_update():
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)

    # ema.average returns the Variable holding the average of var.
    mean, var = tf.cond(is_training,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
  return normed
