#!/bin/sh

## This script only runs 1/3 of all the inputs in two settings, as a functionality verification
## This script takes 3 hours to run

pwd="/home/odroid/Workspace/slambench"

cd $pwd

## Copy the float-precision version of opencl kernel and build the program
cp kfusion/src/opencl/kernels.cl.float kfusion/src/opencl/kernels.cl
cp kfusion/src/opencl/kernels.cpp.float kfusion/src/opencl/kernels.cpp
make

## Run slambench in its default setting
python ae_scripts/run_benchmark_local_power.py default 1 DEFAULT 0 0 simple

## Copy the half-precision version of opencl kernel and build the program
cp kfusion/src/opencl/kernels.cl.half kfusion/src/opencl/kernels.cl
cp kfusion/src/opencl/kernels.cpp.half kfusion/src/opencl/kernels.cpp
make

## Run slambooster (PID + surface detection + pose correction + half precision) 
python ae_scripts/run_benchmark_local_power.py slambooster 1 CTRL_PI 1 1 simple



## Empty the stale figure in figs_simple directory
rm -rf ae_scripts/figs_simple/*

## Plot the performance of slambooster compared to default setting
python3 ae_scripts/data_analysis_condor_plot_summary.py simple default slambooster
