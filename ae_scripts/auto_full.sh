#!/bin/sh

## This script runs all the inputs in different settings, as a full verification.
## The whole experiment takes around 24 hours to finish

pwd="/home/odroid/Workspace/slambench"

cd $pwd

## Copy the float-precision version of opencl kernel and build the program
cp kfusion/src/opencl/kernels.cl.float kfusion/src/opencl/kernels.cl
cp kfusion/src/opencl/kernels.cpp.float kfusion/src/opencl/kernels.cpp
make

## Run slambench in its default setting
python ae_scripts/run_benchmark_local_power.py default 1 DEFAULT 0 0 full

## Run with only PID 
python ae_scripts/run_benchmark_local_power.py pid_only 1 CTRL_PI 0 0 full

## Run with PID + surface detection
python ae_scripts/run_benchmark_local_power.py pid_surface 1 CTRL_PI 1 0 full

## Run with PID + surface detection + pose correction
python ae_scripts/run_benchmark_local_power.py pid_surface_pose 1 CTRL_PI 1 1 full

## Copy the half-precision version of opencl kernel and build the program
cp kfusion/src/opencl/kernels.cl.half kfusion/src/opencl/kernels.cl
cp kfusion/src/opencl/kernels.cpp.half kfusion/src/opencl/kernels.cpp
make

## Run slambooster (PID + surface detection + pose correction + half precision) 
python ae_scripts/run_benchmark_local_power.py slambooster 1 CTRL_PI 1 1 full




## Empty the stale figure in figs_full directory
rm -rf ae_scripts/figs_full/*

## Plot the performance of slambooster compared to default setting
python3 ae_scripts/data_analysis_condor_plot_summary.py full default slambooster

## Plot the incremental performance enhancement
python3 ae_scripts/data_analysis_condor_plot_summary.py full default pid_only pid_surface pid_surface_pose slambooster
