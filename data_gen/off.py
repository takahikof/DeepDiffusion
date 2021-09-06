# -*- coding: utf-8 -*-
import numpy as np

class Mesh :
    def __init__( self, in_filepath, radius=1.0 ) :
        self.n_vert = 0;  # 頂点数
        self.n_face = 0;  # 面数
        # self.n_edge = 0; # 辺数
        self.vert = None; # 各頂点の座標．サイズは(n_vert x 3)．
        self.face = None; # 各面を構成する頂点ID．3角形を想定するのでサイズは(n_face x 3)．
        self.norm_face = None; # 各面の法線ベクトル
        self.norm_vert = None; # 各頂点の法線ベクトル
        self.area_face = None; # 各面の面積
        self.area_total = 0;   # 表面積

        # print( "loading " + in_filepath );

        self.load_off( in_filepath );
        self.calc_norm();
        self.normalize_pos_scale( radius );
        self.calc_area();

    ##### OFF形式ファイルを読み込む #####
    def load_off( self, in_filepath ) :
        f = open( in_filepath );
        data = f.read();  # ファイル終端まで全て読んだデータを返す
        f.close();

        lines = data.split('\n') # 改行で区切る

        headers = lines[ 1 ].split();
        self.n_vert = int( headers[ 0 ] ); # 頂点数
        self.n_face = int( headers[ 1 ] ); # 面数

        # 頂点座標を読み込む
        self.vert = np.zeros( ( self.n_vert, 3 ), dtype=np.float32 );
        for i in range( self.n_vert ) :
            v = lines[ 2 + i ].split();
            self.vert[ i, 0 ] = float( v[ 0 ] );
            self.vert[ i, 1 ] = float( v[ 1 ] );
            self.vert[ i, 2 ] = float( v[ 2 ] );

        # 面を構成する頂点を読み込む
        self.face = np.zeros( ( self.n_face, 3 ), dtype=np.int32 );
        for i in range( self.n_face ) :
            f = lines[ 2 + self.n_vert + i ].split();
            if( int( f[ 0 ] ) != 3 ) :
                print( "error: supports only triangles." );
                quit();
            self.face[ i, 0 ] = int( f[ 1 ] );
            self.face[ i, 1 ] = int( f[ 2 ] );
            self.face[ i, 2 ] = int( f[ 3 ] );

    ##### OFF形式ファイルを書き出す #####
    def save_off( self, out_filepath ) :
        f = open( out_filepath, 'w' );
        f.write( "OFF\n" );
        f.write( str( self.n_vert ) + " " + str( self.n_face ) + " 0\n" );
        for i in range( self.n_vert ) :
            f.write( str( self.vert[ i ][ 0 ] ) + " " + str( self.vert[ i ][ 1 ] ) + " " + str( self.vert[ i ][ 2 ] ) + "\n" );
        for i in range( self.n_face ) :
            f.write( "3 " + str( self.face[ i ][ 0 ] ) + " " + str( self.face[ i ][ 1 ] ) + " " + str( self.face[ i ][ 2 ] ) + "\n" );
        f.close();

    ##### 法線ベクトルを計算  #####
    def calc_norm( self ) :
        # 面の法線を計算
        self.norm_face = np.zeros( ( self.n_face, 3 ), dtype=np.float32 );
        for i in range( self.n_face ) :
            v0 = self.vert[ self.face[ i, 0 ] ];
            v1 = self.vert[ self.face[ i, 1 ] ];
            v2 = self.vert[ self.face[ i, 2 ] ];
            vv1 = v1 - v0;
            vv2 = v2 - v1;
            cross = np.cross( vv1, vv2 );

            norm = np.linalg.norm( cross );
            if( norm < 1e-6 ) :
                normvec = [ 0.0, 0.0, 0.0 ];
            else :
                normvec = cross / norm;
            self.norm_face[ i ] = normvec;

        # 頂点->その頂点が属する面IDリスト
        vert2face = [];
        for i in range( self.n_vert ) :
            vert2face.append( [] );
        for i in range( self.n_face ) :
            vert2face[ self.face[ i, 0 ] ].append( i );
            vert2face[ self.face[ i, 1 ] ].append( i );
            vert2face[ self.face[ i, 2 ] ].append( i );

        # 頂点の法線を計算
        self.norm_vert = np.zeros( ( self.n_vert, 3 ), dtype=np.float32 );
        for i in range( self.n_vert ) :
            normvec = [ 0.0, 0.0, 0.0 ];
            for j in range( len( vert2face[ i ] ) ) :
                normvec += self.norm_face[ vert2face[ i ][ j ] ];

            norm = np.linalg.norm( normvec );
            if( norm < 1e-6 ) :
                normvec = [ 0.0, 0.0, 0.0 ];
            else :
                normvec = normvec / norm;
            self.norm_vert[ i ] = normvec;

        """
        tmp = np.concatenate( ( self.vert, self.norm_vert ), axis = 1 );
        np.savetxt( "out.xyz", tmp );
        """

    ##### 位置とスケールの正規化 #####
    def normalize_pos_scale( self, radius ) :

        # 位置の正規化
        # 頂点群の重心で正規化する場合
        # mean = np.mean( self.vert, axis=0 );
        # self.vert = self.vert - mean;

        # 頂点群のbounding boxで正規化する場合
        bbcenter = ( np.max( self.vert, 0 ) + np.min( self.vert, 0 ) ) / 2.0;
        self.vert = self.vert - bbcenter;

        # スケールの正規化 (半径radiusの球に収まるようにスケーリング)
        norms = np.linalg.norm( self.vert, axis=1 );
        max_norm = np.max( norms );
        self.vert = self.vert * ( radius / max_norm );

    ##### 面の重心を計算 #####
    def calc_face_gravity_center( self ) :
        face_gc = np.zeros( ( self.n_face, 3 ), dtype=np.float32 );
        for i in range( self.n_face ) :
            mean = [ 0.0, 0.0, 0.0 ];
            for j in range( 3 ) :
                mean += self.vert[ self.face[ i, j ] ];
            face_gc[ i ] = mean / 3.0;
        return face_gc;

    ##### 面の面積を計算 #####
    def calc_area( self ) :
        self.area_face = np.zeros( self.n_face, dtype=np.float32 );
        self.area_total = 0.0;
        for i in range( self.n_face ) :
            a = self.vert[ self.face[ i, 0 ] ];
            b = self.vert[ self.face[ i, 1 ] ];
            c = self.vert[ self.face[ i, 2 ] ];
            v1 = a - b;
            v2 = c - b;
            tmp = np.cross( v1, v2 );
            tmp = 0.5 * np.linalg.norm( tmp );
            self.area_face[ i ] = tmp;
            self.area_total += tmp;



if( __name__ == "__main__" ) :

    mesh = Mesh( "icosahedron.off" );
    quit();
