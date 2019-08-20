import os, sys, time
import socket
import subprocess
import telnetlib

trials = int(sys.argv[2])
heuristic = sys.argv[3]
if_checkwall = sys.argv[4]
if_extrapolate = sys.argv[5]
dataset_option = sys.argv[6]

SMART_POWER_HOST = "192.168.4.1"

### Specify the input ###
hostname = socket.gethostname()
if hostname == "torque.ices.utexas.edu":
    input_dir = "/net/ohm/export/cdgc/ypei/approx-applications/slam/icl-nuim-raw/"
    output_prefix = "/workspace/ypei/capri-output/slam/" + sys.argv[1]
    executable_path = "/workspace/ypei/slambench3/slambench/build/kfusion/kfusion-benchmark-opencl"
elif hostname == "austin-gd-odroid0":
    input_dir = "/home/ypei/Research/slam/slam_input/"
    output_prefix = "/home/ypei/Research/slam/slam_output/" + sys.argv[1]
    executable_path = "/home/ypei/Research/slam/slambench/build/kfusion/kfusion-benchmark-opencl"
elif hostname == "odroid-AE":
    input_dir = "/home/odroid/Workspace/slam_inputs/"
    output_prefix = "/home/odroid/Workspace/slam_outputs/" + sys.argv[1]
    executable_path = "/home/odroid/Workspace/slambench/build/kfusion/kfusion-benchmark-opencl"
else:
    print "Unknown hostname"
    sys.exit()

column_list = ['frame', 'acquistion', 'control', 'preprocessing', 'tracking', 'integration',
               'raycasting', 'rendering', 'computation', 'totaltime',
               'xt', 'yt','zt', 'vxt', 'vyt', 'vzt', 'axt', 'ayt', 'azt',
               'iftracked', 'ifintegrated', 'csr', 'icp', 'ir',
               'frame_diff', 'if_wall', 'if_jerky',
               'pred_x', 'pred_y', 'pred_z',
              ]
num_columns = len(column_list)

benchmark_set = [
        "living_room_traj0_loop", "5.5", "0.25,0.5,0.25",
        "living_room_traj1_loop", "5.5", "0.5,0.5,0.6",
        "living_room_traj2_loop", "5.5", "0.35,0.5,0.35",
        "living_room_traj3_loop", "5.5", "0.3,0.5,0.45",

        "postdoc_office0_traj_loop", "6", "0.65,0.5,0.5",
        "postdoc_office1_traj_loop", "6", "0.60,0.5,0.45",

        "kitchen0_traj", "4.5", "0.5,0.5,0.35",
        "kitchen1_traj", "4.5", "0.5,0.5,0.35",
        "lab0_traj", "4", "0.5,0.5,0.35",
        "lab1_traj", "5", "0.5,0.5,0.35",
        "lab2_traj", "5", "0.5,0.5,0.35",
        "lab3_traj", "5", "0.5,0.5,0.35",
        "meeting_room0_traj", "5.5", "0.6,0.5,0.35",
        "meeting_room1_traj", "5.5", "0.45,0.5,0.35",
        "meeting_room2_traj", "5.5", "0.55,0.5,0.35",
        "office0_traj", "5", "0.75,0.5,0.35",
        "office1_traj", "5", "0.25,0.5,0.35",
        "office2_traj", "5", "0.5,0.5,0.35",

        "rgbd_dataset_freiburg1_xyz", "4", "0.5,0.5,0.15",
        "rgbd_dataset_freiburg1_floor", "4", "0.5,0.5,0.15",
        "washington_01", "5", "0.5,0.5,0.25",
        "washington_03", "5", "0.5,0.5,0.25",
        "washington_08", "5", "0.5,0.5,0.25",
        "washington_13", "5", "0.5,0.5,0.15",
        "washington_14", "5", "0.5,0.5,0.15"
    ]

### Specify the benchmark ###

if dataset_option == "simple":
    testing_benchmark_set = ["living_room_traj0_loop",
                             "postdoc_office1_traj_loop",
                             "kitchen0_traj",
                             "lab2_traj",
                             "meeting_room1_traj",
                             "office0_traj"
                             ]

elif dataset_option == "full":
    testing_benchmark_set = [
                             "living_room_traj0_loop",
                             "living_room_traj1_loop",
                             "living_room_traj2_loop",

                             "postdoc_office0_traj_loop",
                             "postdoc_office1_traj_loop",

                             "kitchen0_traj",
                             "kitchen1_traj",

                             "lab0_traj",
                             "lab1_traj",
                             "lab2_traj",
                             "lab3_traj",

                             "meeting_room0_traj",
                             "meeting_room1_traj",
                             "meeting_room2_traj",

                             "office0_traj",
                             "office1_traj",
                             "office2_traj"
                             ]
else:
    print "Only two options allowed for dataset: [simple, full]"
    sys.exit()


num_benchmarks = len(testing_benchmark_set)

### Specify each knob to start with ###
csr = 1
icp = 0
mu = 0.1

def main():

    job_counter = 0;

    for trial in range(trials):
        for benchmark in testing_benchmark_set:
            path_appendix =('/trial=' + str(trial)
                            + '/benchmark=' + benchmark
                            + '/'
                              )
            output_dir = output_prefix + path_appendix
            script_dir = output_prefix + "/scripts" + path_appendix

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if not os.path.exists(script_dir):
                os.makedirs(script_dir)

            index = benchmark_set.index(benchmark)
            vol_size = benchmark_set[index+1]
            init_pose = benchmark_set[index+2]

            cmd = (executable_path
                + " -i " + input_dir + benchmark
                + " -s " + vol_size
                + " -p " + init_pose
                + " -c " + str(csr)
                + " -l " + str(icp)
                + " -m " + str(mu)
                + " -z 1"
                + " -r 1"
                + " -t 1"
                + " -v 256,256,256"
                + " -k 481.2,480,320,240"
                + " -y 10,5,4"
                + " -u " + heuristic
                + " -w " + if_checkwall
                + " -x " + if_extrapolate
                + " -o " + output_dir + "output.log"
                )


            f_script = open(script_dir + benchmark + ".sh", 'w')
            f_script.write(cmd)
            f_script.close()
            os.system("chmod +x " + script_dir + benchmark + ".sh")

            time.sleep(10) #Time interval between benchmarks
            tn = telnetlib.Telnet(SMART_POWER_HOST)
            #tn.set_debuglevel(1)
            #tn.msg("running " + benchmark)

            current_pwd = os.getcwd()
            os.chdir(script_dir)
            os.system("./" + benchmark + ".sh")
            os.chdir(current_pwd)

            power_log = tn.read_very_eager()
            tn.close()

            # print power_log
            f_power = open(output_dir + "energy.log", "w")
            f_power.write(power_log)
            f_power.close()

            job_counter += 1

    print "total job =", job_counter


if __name__ == "__main__":
    main()
