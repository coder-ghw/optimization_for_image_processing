#!/bin/bash
echo "Starting the application..."
do_build=$1
if [ $do_build == 1 ]; then
    echo "Building the application..."
    if [ ! -d "build" ]; then
    echo "Build directory does not exist. Creating build directory..."
    mkdir build
    fi

    cd build
    cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
    make -j8
else
    cd build
    echo "Skipping build step."
fi

echo "Application started."

file_img="/home/guohw/projects/optimization/data/000000000419.jpg"
 ./train_kernel --img $file_img\
                --optimizer gn\
                --damping 1e-6\
                --lr 1.0\
                --iters 10

./train_kernel --img $file_img\
               --optimizer adam \
               --lr 0.001 \
               --reg 1e-4 \
               --iters 200

