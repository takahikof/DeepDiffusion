# -*- coding: utf-8 -*-
import numpy as np
import time
import random
import math
import sys
import os

from sklearn.metrics import pairwise_distances
from sklearn.metrics import label_ranking_average_precision_score
from scipy.sparse import csr_matrix
from scipy.stats import rankdata

def retrieval( embedding, label, calc_rpcurve=True ) :

    recall_step = 0.05;

    mean_average_precision = 0.0;
    mean_recall = np.zeros( int( 1.0 / recall_step ) );
    mean_precision = np.zeros( int( 1.0 / recall_step ) );

    n_data = embedding.shape[0];

    D = pairwise_distances( embedding, metric="cosine" );

    for i in range( n_data ) : # for each query

        dist_vec = D[ i ];
        gt_vec = np.asarray( label==label[i], dtype=np.int32 ); # 1 if the retrieval target belongs to the same category with the query, 0 otherwise

        dist_vec_woq = np.delete( dist_vec, i ); # distance vector without query
        gt_vec_woq = np.delete( gt_vec, i );     # groundtruth vector without query
        gt_vec_woq_sp = csr_matrix( gt_vec_woq );   # convert to sparse matrix

        relevant = gt_vec_woq_sp.indices;
        n_correct = gt_vec_woq_sp.nnz; # number of correct targets for the query
        rank = rankdata( dist_vec_woq, 'max')[ relevant ]; # positions where correct data appear in a retrieval ranking
        rank_sorted = np.sort( rank );

        # average precision
        if( n_correct == 0 ) :
            ap = 1.0;
        else :
            L = rankdata( dist_vec_woq[relevant], 'max');
            ap = (L / rank).mean();
        mean_average_precision += ap;

        # recall-precision curve
        if( calc_rpcurve ) :
            one_to_n = ( np.arange( n_correct ) + 1 ).astype( np.float32 );
            precision = one_to_n / rank_sorted;
            recall = one_to_n / n_correct;
            recall_interp = np.arange( recall_step, 1.01, recall_step );
            precision_interp = np.interp( recall_interp, recall, precision );
            mean_recall = recall_interp; # no need to average
            mean_precision += precision_interp;

    mean_average_precision /= n_data;

    if( calc_rpcurve ) :
        mean_precision /= n_data;
    else :
        mean_recall = None;
        mean_precision = None;

    return( mean_average_precision, ( mean_recall, mean_precision ) );
