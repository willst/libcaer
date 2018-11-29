#!/usr/bin/env bash

source activate dvs
export PKG_CONFIG_PATH=${CONDA_ENV_PATH}/lib/pkgconfig
g++ -std=c++11 -fPIC -shared -Wno-undef -O3 -I/usr/local/include/pybind11 -I${CONDA_ENV_PATH}/lib/python2.7/site-packages/numpy/core/include $(pkg-config --cflags --libs python2 eigen3 opencv) dvs128_py.cpp -o optic_flow.so
