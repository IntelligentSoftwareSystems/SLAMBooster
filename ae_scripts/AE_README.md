# README #

This README is specific for the artifact evaluation process of PACT19

General information:
    Program: Slambench /home/odroid/Workspace/slambench
    Trajectory inputs: 17 inputs in /home/odroid/Workspace/slam_inputs
    Runtime stats outputs: /home/odroid/Workspace/slam_outputs
    AE scripts: /home/odroid/Workspace/slambench/ae_scripts
    AE figure outputs: /home/odroid/Workspace/slambench/ae_scripts/fig_simple
                       /home/odroid/Workspace/slambench/ae_scripts/fig_full


Script guide (/home/odroid/Workspace/slambench/ae_scripts):
    auto_simple.sh: it verifies the functionality of the program.
                    This script will run 6 inputs in two algorithm settings(default and slambooster)
                    Plots will be generated in directory: /home/odroid/Workspace/slambench/ae_scripts/fig_simple
                    Please use -XC to ssh or scp figures to other places for review.
        How to run: ./auto_simple.sh

    auto_full.sh: it runs the full experiment with different settings described in Seciont V-B and V-C.
                  Plots will be generated in directory: /home/odroid/Workspace/slambench/ae_scripts/fig_full
                  Please use -XC to ssh or scp figures to other places for review.
        How to run: ./auto_full.sh

