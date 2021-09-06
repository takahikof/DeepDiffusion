# -*- coding: utf-8 -*-
import numpy as np
import sys
import time
import random
import math
import os
import h5py
import copy

sys.path.append("../utility")
sys.path.append("../evaluation")
import retrieval

from sklearn.metrics import pairwise_distances as np_pairwise_distances
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix


def augment_feature( f_train, f_test, knn=100, alpha=0.99, n_iter=20, knn_for_manifold=20 ) :
    augfeat_test = feataug_by_dm( f_train, f_test, knn, alpha, n_iter, knn_for_manifold );
    return augfeat_test;

def get_neighbor_features( f_train, f_test, knn ) :

    n_data_test = f_test.shape[0];
    metric = "cosine";
    D = np_pairwise_distances( f_test, f_train, metric=metric );

    idx = np.argpartition( D, knn )[:,:knn]; # 値の小さい方からknn個のインデックスを得る．
    idx = idx.tolist();

    neighbor_features = [];
    for i in range( n_data_test ) :
        neighbor_features.append( f_train[ idx[ i ] ] );

    return neighbor_features, idx;

def feataug_by_dm( f_train, f_test, knn, alpha, n_iter, knn_for_manifold ) :

    n_data_test = f_test.shape[0];
    n_dim_feat = f_test.shape[1];
    neighbor_features, neighbor_indices = get_neighbor_features( f_train, f_test, knn );

    #print( "constructing a manifold graph..." );
    n_data_train = f_train.shape[0];
    # metric = "cosine";
    # D = np_pairwise_distances( f_train, f_train, metric=metric ); # memory footprint is too large
    D = 1.0 - ( ( np.dot( f_train, np.transpose( f_train ) ) + 1.0 ) / 2.0 ); # distance is scaled in the range [0,1]

    for i in range( n_data_train ) : # remove self-loop
        D[ i, i ] = 1e10;

    # for efficient diffusion process, the feature manifold graph is represented as a sparse matrix

    # idx = np.argpartition( D, knn_for_manifold )[:,:knn_for_manifold ]; # 値の小さい方からknn個のインデックスを得る．
    # memory consumption of np.argpartition is very large when the input matrix is large (e.g., 100k x 100k).
    # therefore, the matrix is split into three submatrices and np.argpartition is applied to each of the submatrices.
    part1 = int( n_data_train / 3 );
    part2 = 2 * part1;
    idx1 = np.argpartition( D[0:part1], knn_for_manifold )[:,:knn_for_manifold ].copy().astype(np.int32);
    idx2 = np.argpartition( D[part1:part2], knn_for_manifold )[:,:knn_for_manifold ].copy().astype(np.int32);
    idx3 = np.argpartition( D[part2:], knn_for_manifold )[:,:knn_for_manifold ].copy().astype(np.int32);
    idx = np.vstack( [ idx1, idx2, idx3 ] );
    del idx1;
    del idx2;
    del idx3;
    idx = np.sort( idx, axis=1 ); # index順にソート (疎行列を作るため)

    sim_list = [];
    row_list = [];
    col_list = [];
    Diag = [];
    for i in range( n_data_train ) :
        sum_of_row = 0.0;
        for j in range( knn_for_manifold ) :
            similarity = 1.0 - D[ i, idx[ i, j ] ];
            sim_list.append( similarity );
            row_list.append( i );
            col_list.append( idx[ i, j ] );
            sum_of_row += similarity;

        Diag.append( 1.0 / np.sqrt( sum_of_row ) );

    A = csr_matrix( ( sim_list, ( row_list, col_list ) ), ( n_data_train, n_data_train ) );
    Diag = csr_matrix( ( Diag, ( list(range(n_data_train)), list(range(n_data_train)) ) ), ( n_data_train, n_data_train ) );
    S = ( Diag.dot( A ) ).dot( Diag ); # S = D^(-1/2) * A * D^(-1/2)

    del D;
    del A;
    del Diag;

    # prepare diffusion sources
    Sources = [];
    for i in range( n_data_test ) :
        s = np.zeros( ( n_data_train ), dtype=np.float32 );
        s[ neighbor_indices[ i ] ] = 1.0;
        Sources.append( np.reshape( s, (-1,1) ) );
    Sources = np.hstack( Sources );
    F = copy.deepcopy( Sources );

    # iterate the diffusion process
    for i in range( n_iter ) :
        # print( i )
        F = alpha * S.dot( F ) + ( 1.0 - alpha ) * Sources;

    F = np.transpose( F );

    # power normalization
    F = np.sign( F ) * np.sqrt( np.abs( F ) );
    norm = np.reshape( np.linalg.norm( F, axis=1 ), ( -1, 1 ) );
    norm[ norm < 1e-6 ] = 1e-6;
    F = F / np.tile( norm, ( 1, n_data_train ) );

    augfeat_test = F;

    augfeat_test = np.hstack( [ f_test, augfeat_test ] );

    return augfeat_test;
