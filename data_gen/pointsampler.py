# -*- coding: utf-8 -*-
import numpy as np
import sobol
from off import *

class PointSampler :
    def __init__( self, n_point ) :
        self.n_point = n_point;
        self.seed = 1;

    def sample( self, mesh ) :
        pos = np.zeros( ( self.n_point, 3 ), dtype=np.float32 );
        ori = np.zeros( ( self.n_point, 3 ), dtype=np.float32 );

        sa = mesh.area_total / float( self.n_point );  # 点を1個生成する面積

        error = 0.0;
        count = 0;

        for i in range( mesh.n_face ) :
            tmp = mesh.area_face[ i ]; # 面iの面積

            # 面iにまく点の数nptを計算
            npt = 0;
            tmp /= sa;
            tmp2 = tmp;
            while( tmp >= 1 ) : # 対象の三角形に何点生成するか
                npt += 1;
                tmp -= 1;
            error += tmp2 - npt;  # 小数部分をまとめる
            if( error >= 1 ) :    # 小数部分が1より大きければ生成する点数を増加
                npt += 1;
                error -= 1;

            for j in range( npt ) : # 各点の座標値を計算
                vec, self.seed = sobol.i4_sobol( 2, self.seed );
                r1 = np.sqrt( vec[ 0 ] );
                r2 = vec[ 1 ];
                for k in range( 3 ) : # Osadaの方法
                    pos[ count ][ k ] = \
                    ( 1.0-r1 ) * mesh.vert[ mesh.face[ i ][ 0 ] ][ k ] + \
                    r1 * ( 1.0 - r2 ) * mesh.vert[ mesh.face[ i ][ 1 ] ][ k ] + \
                    r1 * ( r2 * mesh.vert[ mesh.face[ i ][ 2 ] ][ k ] );
                ori[ count ] = mesh.norm_face[ i ]; # 点の向きは面の法線
                count += 1;

        if( count != self.n_point ) :  # 生成する点の数が不足していたら追加
            vec, self.seed = sobol.i4_sobol( 2, self.seed );
            r1 = np.sqrt( vec[ 0 ] );
            r2 = vec[ 1 ];
            for k in range( 3 ) : # Osadaの方法
                pos[ self.n_point - 1 ][ k ] = \
                ( 1.0-r1 ) * mesh.vert[ mesh.face[ mesh.n_face - 1 ][ 0 ] ][ k ] + \
                r1 * ( 1.0 - r2 ) * mesh.vert[ mesh.face[ mesh.n_face - 1 ][ 1 ] ][ k ] + \
                r1 * ( r2 * mesh.vert[ mesh.face[ mesh.n_face - 1 ][ 2 ] ][ k ] );
            ori[ self.n_point - 1 ] = mesh.norm_face[ mesh.n_face - 1 ]; # 点の向きは面の法線
            count += 1;

        return np.hstack( [ pos, ori ] );


if( __name__ == "__main__" ) :
    mesh = Mesh( "T0.off" );
    pointsampler = PointSampler( 2048 );
    oript = pointsampler.sample( mesh );
    np.savetxt( "out.xyz", oript );
    quit();
