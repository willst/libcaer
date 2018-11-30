#!/usr/bin/env bash

#Handled now on RPi in ~/.bashrc

#source activate dvs
#export PKG_CONFIG_PATH=${CONDA_ENV_PATH}/lib/pkgconfig
#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_ENV_PATH}/lib


g++ -std=c++11 -fPIC -shared -Wno-undef -O3 -D_DEFAULT_SOURCE=1 -I/usr/local/include/pybind11 $(pkg-config --cflags python2 eigen3 opencv libcaer) $(pkg-config --libs libcaer) dvs128_py.cpp -o optic_flow.so
