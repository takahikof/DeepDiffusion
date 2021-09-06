#!/bin/bash

# 1. Download the ModelNet10 dataset
echo "downloading dataset files..."
dataset="http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"

tmp_dir="./tmp_modelnet10"
mkdir -p $tmp_dir
wget $dataset
mv ModelNet10.zip $tmp_dir

unzip $tmp_dir/ModelNet10.zip -d $tmp_dir

# 2. Move all the OFF files (the polygonal 3D models) into a directory
echo "aggregating OFF files into a directory..."
off_dir=$tmp_dir/"off"
mkdir -p $off_dir
find $tmp_dir/ModelNet10 -type f -name "*.off" | while read i;
do
  mv $i $off_dir
done

# 3. Convert polygonal 3D shapes to 3D point set shapes and write them to HDF5 files
echo "converting polygonal 3D shapes to 3D point set shapes..."
echo "this will take a long time due to the surface sampling process."
data_dir="./data"
n_points=2048
mkdir -p $data_dir
python data_gen/GenerateH5_ModelNet10.py $off_dir $n_points data_gen/label_ModelNet10_train.txt data_gen/ctglist_modelnet10.txt data/modelnet10_train.h5
python data_gen/GenerateH5_ModelNet10.py $off_dir $n_points data_gen/label_ModelNet10_test.txt data_gen/ctglist_modelnet10.txt data/modelnet10_test.h5

# 4. Delete temporary files
rm -rf $tmp_dir

exit
