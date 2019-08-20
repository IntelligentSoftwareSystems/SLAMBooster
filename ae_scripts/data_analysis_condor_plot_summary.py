import os, sys
import numpy as np
import math
import socket
import subprocess
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib.ticker import FormatStrFormatter

hostname = socket.gethostname()
output_prefix = ""
if hostname == "torque.ices.utexas.edu":
    output_prefix = "/workspace/ypei/capri-output/slam/"
    gt_file_prefix = "/workspace/ypei/slambench3/slambench/gt_new_031918/"
    time_yrange = [0.0, 0.4]
elif hostname == "austin-gd-odroid0":
    output_prefix = "/home/ypei/Research/slam/slam_output/"
    gt_file_prefix = "/home/ypei/Research/slam/slambench/gt_new_031918/"
    time_yrange = [0.0, 0.35]
elif hostname == "odroid-AE":
    output_prefix = "/home/odroid/Workspace/slam_outputs/"
    gt_file_prefix = "/home/odroid/Workspace/slambench/gt_new_031918/"
    time_yrange = [0.0, 0.35]

else:
    print("Unknown host")
    sys.exit()

column_list = ['frame', 'acquistion', 'control', 'preprocessing', 'tracking', 'integration',
               'raycasting', 'rendering', 'computation', 'total',
               'xt', 'yt','zt', 'vxt', 'vyt', 'vzt', 'vt', 'axt', 'ayt', 'azt',
               'iftracked', 'ifintegrated', 'csr', 'icp', 'ir', 'vr',
               'if_wall', 'if_jerky',
               'pred_x', 'pred_y', 'pred_z',
              ]
num_columns = len(column_list)

POS_INDEX = column_list.index('xt')
VELOCITY_INDEX = column_list.index('vxt')
COMPUTATION_TIME = column_list.index('computation')
TOTAL_TIME = column_list.index('total')
CTRL_TIME = column_list.index('control')
IF_TRACKED = column_list.index('iftracked')
IF_INTEGRATED = column_list.index('ifintegrated')
CSR_INDEX = column_list.index('csr')
ICP_INDEX = column_list.index('icp')
IR_INDEX = column_list.index('ir')
IF_WALL_INDEX = column_list.index('if_wall')
IF_JERKY_INDEX = column_list.index('if_jerky')
PRED_TRANS_INDEX = column_list.index('pred_x')

### Specify the benchmark ###
dataset_option = sys.argv[1]

if dataset_option == "simple":
    testing_benchmark_set = ["living_room_traj0_loop",
                             "kitchen0_traj",
                             "lab2_traj",
                             "meeting_room1_traj",
                             "office0_traj",
                             "postdoc_office1_traj_loop"
                             ]
    testing_benchmark_labels = ['lr0','ktcn0',
                                'lab2','mr1',
                                'off0','pd1',
                                'geomean']

elif dataset_option == "full":
    testing_benchmark_set = [
                             "living_room_traj0_loop",
                             "living_room_traj1_loop",
                             "living_room_traj2_loop",

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
                             "office2_traj",

                             "postdoc_office0_traj_loop",
                             "postdoc_office1_traj_loop"
                             ]
    testing_benchmark_labels = ['lr0', 'lr1', 'lr2', 'ktcn0', 'ktcn1',
                                'lab0', 'lab1', 'lab2', 'lab3', 'mr0', 'mr1', 'mr2',
                                'off0', 'off1', 'off2',
                                'pd0', 'pd1',
                                'geomean']


else:
    print("Only two options allowed for dataset: [simple, full]")
    sys.exit()


num_benchmarks = len(testing_benchmark_set)

### Specify each knob to start with ###
csr = 1
icp = 0
mu = 0.1
idle_power = 4.0

def main():
    if len(sys.argv) == 2:
        print("Please provide at least one output directory part")
        sys.exit()

    num_targets = len(sys.argv) - 2;
    target_dir_list = []
    for argv in sys.argv[2:]:
        target_dir_list.append(output_prefix + argv)
    assert(num_targets == len(target_dir_list))

    energy_flag = True
    result_dict = {}
    for result_prefix in target_dir_list:
        print(result_prefix)

        num_trials = 0
        for name in os.listdir(result_prefix):
            if name[0:len("trial=")] == "trial=":
                num_trials += 1

        if not os.path.exists(result_prefix + '/summary'):
            os.makedirs(result_prefix + '/summary')

        result_dict[result_prefix] = {}
        result_dict[result_prefix]['csr'] = []
        result_dict[result_prefix]['icp'] = []
        result_dict[result_prefix]['ir'] = []
        result_dict[result_prefix]['vt'] = []
        result_dict[result_prefix]['if_wall'] = []
        result_dict[result_prefix]['if_jerky'] = []
        result_dict[result_prefix]['err'] = []
        result_dict[result_prefix]['comptime'] = []
        result_dict[result_prefix]['totaltime'] = []
        result_dict[result_prefix]['tracking_rate'] = []
        result_dict[result_prefix]['first_tracking_rate'] = []
        result_dict[result_prefix]['energy'] = []

        for benchmark in testing_benchmark_set:
            gt_file = gt_file_prefix + benchmark + '.gt'

            '''
            csr_list
            icp_list
            ir_list
            vt_list
            if_wall_list
            if_jerky_list
            err_list
            comptime_list
            totaltime_list
            '''
            len_results = 9
            all_results = []
            total_energy = 0

            for trial in range(num_trials):
                path_appendix =('/trial=' + str(trial)
                               + '/benchmark=' + benchmark
                               # + '/input=' + benchmark
                               + '/'
                               )

                log_file = result_prefix + path_appendix + 'output.log'
                (trial_results, lost_list, lost_err_list,
                        first_tracking_rate, tracking_rate) = process(log_file, gt_file)

                energy_file = result_prefix + path_appendix + 'energy.log'
                trial_energy = process_energy(energy_file, len(trial_results[0]))
                if energy_flag:
                    if trial_energy == -1:
                        energy_flag = False
                        print("No energy log file available " + energy_file)
                    else:
                        total_energy += trial_energy

                if trial == 0:
                    for i in range(len_results):
                        all_results.append(trial_results[i])
                else:
                    for i in range(len_results - 2):
                        assert(all_results[i] == trial_results[i])
                    assert(len(all_results[-1]) == len(trial_results[-1]))
                    assert(len(all_results[-2]) == len(trial_results[-2]))
                    for i in range(len(all_results[-1])):
                        all_results[-1][i] += trial_results[-1][i]
                        all_results[-2][i] += trial_results[-2][i]

            result_dict[result_prefix]['csr'].append(np.mean(all_results[0]))
            result_dict[result_prefix]['icp'].append(np.mean(all_results[1]))
            result_dict[result_prefix]['ir'].append(np.mean(all_results[2]))
            result_dict[result_prefix]['vt'].append(np.mean(all_results[3]))
            result_dict[result_prefix]['if_wall'].append(np.mean(all_results[4]))
            result_dict[result_prefix]['if_jerky'].append(np.mean(all_results[5]))
            result_dict[result_prefix]['err'].append(np.mean(all_results[6]))
            result_dict[result_prefix]['comptime'].append(np.mean(all_results[7])/num_trials)
            result_dict[result_prefix]['totaltime'].append(np.mean(all_results[8])/num_trials)

            result_dict[result_prefix]['tracking_rate'].append(tracking_rate)
            result_dict[result_prefix]['first_tracking_rate'].append(first_tracking_rate)
            result_dict[result_prefix]['energy'].append(total_energy/num_trials)


    #num_xticks = num_benchmarks
    num_xticks = num_benchmarks + 1
    frange_ticks = list(range(num_xticks))
    frange = []
    for i in range(num_targets):
        frange.append(list(range(num_xticks)))

    width = 1.0/(num_targets + 1)
    for i in range(num_targets):
        for j in range(num_xticks):
            frange[i][j] += width*(i + 1/2.0)

    for j in range(num_xticks):
        frange_ticks[j] += (num_targets)*width/2

    #incremental = 1
    incremental = (num_targets > 2) 
    if incremental:
        #color_index = ['SkyBlue', 'Pink', "Green", "Orange", "Cyan", "Blue"] #for colored output
        #hatch_index = ["", "", "", "", ""]
        color_index = ['k', "dimgray", "dimgray", "lightgray", "lightgray"] #for paper opt
        hatch_index = ["", "//////", "", "\\\\\\\\\\\\", ""]
        plot_legend_list = ['DEFAULT', 'PID Controller', 'PID Controller + Detection',
                            'PID Controller + Detection + Correction', 'SLAMBooster']
        plot_yrange_list = [[0, 15.0001], time_yrange, [0, 3.0001]] #for incremental impact
        plot_ystep_list = [5.0, 0.1, 1] #for incremental impact
    else:
        color_index = ['k', 'lightgray']
        hatch_index = ["", "", "", "", ""]
        plot_legend_list = ['DEFAULT', 'SLAMBooster']
        plot_yrange_list = [[0, 7.0], time_yrange, [0, 3.0]] #for odroid results
        plot_ystep_list = [2.0, 0.1, 1] #for odroid results

    plot_key_list = ['err', 'comptime']
    if energy_flag:
        plot_key_list.append('energy')
    plot_title_list = ['Average Trajectory Error',
                       'Computation time per frame',
                       'Energy per frame']
    plot_ylabel_list = ['ATE (cm)', 'Computation Time (s)', 'Energy (Joule)']

    fig_dir = "/home/odroid/Workspace/slambench/ae_scripts/figs_" + dataset_option + "/"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    for i in range(len(plot_key_list)):
        print("\n\n")
        for j in range(num_targets):
            tmp_nparray = np.array(result_dict[target_dir_list[j]][plot_key_list[i]])
            geomean = tmp_nparray.prod()**(1.0/len(tmp_nparray))
            result_dict[target_dir_list[j]][plot_key_list[i]].append(geomean)
            label = target_dir_list[j].split('/')[-1]

            print("-------------------------------------------------------------------------------")
            print(plot_key_list[i], label)
            for k in range(num_xticks):
                print(testing_benchmark_labels[k],
                        result_dict[target_dir_list[j]][plot_key_list[i]][k])


        #fig, ax = plt.subplots(figsize=(16, 9))
        fig, ax = plt.subplots(figsize=(12, 1.8)) # for paper graph generation
        if plot_key_list[i] == "err":
            plt.axhline(y=5, color='grey', linestyle='--')
        elif plot_key_list[i] == "comptime":
            plt.axhline(y=0.10, color='grey', linestyle='--')

        rect = []
        for j in range(num_targets):

            if len(plot_legend_list) != 0:
                label_str = plot_legend_list[j]
            else:
                label_str = target_dir_list[j].split('/')[-1]

            rect.append(ax.bar(frange[j],
                        result_dict[target_dir_list[j]][plot_key_list[i]],
                        width,
                        color=color_index[j],
                        hatch=hatch_index[j],
                        label=label_str)
            )

        ax.set_title(plot_title_list[i])
        ax.set_ylabel(plot_ylabel_list[i])
        ax.set_xticks(frange_ticks)
        ax.set_xticklabels(testing_benchmark_labels)
        ax.set_ylim(plot_yrange_list[i])
        ax.set_yticks(np.arange(plot_yrange_list[i][0],
                                plot_yrange_list[i][1],
                                plot_ystep_list[i]))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        if incremental:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.85])
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=5,
                    fontsize="small", frameon=False)
        else:
            ax.legend(fontsize="small", loc="upper right", ncol=2)

        plt.tight_layout()
        ax.margins(0.025)
        if incremental == 0:
            plt.savefig(fig_dir + plot_key_list[i] + '.pdf')
        else:
            plt.savefig(fig_dir + plot_key_list[i] + '_incremental.pdf')
        #plt.show()

        plt.close()



    ## For plotting the comparison between different controllers
    compare_controllers = 0
    if compare_controllers:

        color_index = ['k', 'lightgray']
        plot_key_list = ['err', 'comptime', 'energy']
        plot_ylabel_list = ['ATE (cm)', 'Computation Time per Frame (s)',
                'Energy per Frame (Joule)']
        plot_yrange_list = [[0, 7], [0, 0.11], [0, 1.50001]] #for odroid results
        plot_ystep_list = [2, 0.05, 0.5] #for odroid results
        plot_controller_xlabels = ["Step", "PID"]

        for i in range(len(plot_key_list)):

            fig, ax = plt.subplots(figsize=(2, 2.75))
            if plot_key_list[i] == "err":
                plt.axhline(y=5, color='grey', linestyle='--')
            elif plot_key_list[i] == "comptime":
                plt.axhline(y=0.10, color='grey', linestyle='--')

            plot_data = []
            for j in range(num_targets):
                plot_data.append(result_dict[target_dir_list[j]][plot_key_list[i]][-1])

            frange = np.arange(num_targets)
            plt.bar(frange, plot_data,
                    align='center',
                    color=color_index,
                    width=0.75
                    )

            plt.xticks(frange, plot_controller_xlabels)
            plt.ylabel(plot_ylabel_list[i])

            plt.ylim(plot_yrange_list[i])
            plt.yticks(np.arange(plot_yrange_list[i][0],
                                 plot_yrange_list[i][1],
                                 plot_ystep_list[i]))
            #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            #plt.savefig("./ypei_script/" + plot_key_list[i] + '_ctrl.pdf')
            plt.tight_layout()
            ax.margins(0.025)
            plt.show()

            plt.close()



def process(log_file, gt_file):

    f = open(log_file)
    gt_f = open(gt_file)
    lines = f.readlines()
    gt_lines = gt_f.readlines()
    f.close()
    gt_f.close()

    # first line with tags and first data line whose 'tracked' is false
    lines = lines[2:]  # First four frames are left out in KFusion
    first_gt_words = gt_lines[0].split(' ')
    gt_lines = gt_lines[4:]
    num_lines = min(len(lines), len(gt_lines))

    first_lost = num_lines
    lost_list = []
    for i in range(num_lines):
        line = lines[i]
        words = line.split('\t')
        if words[IF_TRACKED] == "0":  # Not tracked
            if first_lost == num_lines:
                first_lost = i
            lost_list.append(i)

    first_tracking_rate = first_lost/float(num_lines)
    tracking_rate = 1 - len(lost_list)/float(num_lines)
    if len(lost_list):
        print('\t'+log_file.split('/')[-2], num_lines, first_lost, len(lost_list))

    tracked_list = []
    csr_list = []
    icp_list = []
    ir_list = []
    vt_list = []
    if_wall_list = []
    if_jerky_list = []
    err_list = []
    comptime_list = []
    totaltime_list = []
    for i in range(num_lines):
        words = lines[i].split('\t')
        gt_words = gt_lines[i].split(' ')

        tracked_list.append(float(words[IF_TRACKED]))

        vxt = float(words[VELOCITY_INDEX])
        vyt = float(words[VELOCITY_INDEX + 1])
        vzt = float(words[VELOCITY_INDEX + 2])
        vt = math.sqrt(vxt*vxt + vyt*vyt + vzt*vzt)
        vt_list.append(vt)

        csr_list.append(float(words[CSR_INDEX]))
        icp_list.append(float(words[ICP_INDEX]))
        ir_list.append(float(words[IF_INTEGRATED]))

        err_x = float(words[POS_INDEX]) + float(first_gt_words[1]) - float(gt_words[1])
        err_y = float(words[POS_INDEX + 1]) - (float(first_gt_words[2]) - float(gt_words[2]))
        err_z = float(words[POS_INDEX + 2]) + float(first_gt_words[3]) - float(gt_words[3])
        err = math.sqrt(err_x * err_x + err_y * err_y + err_z * err_z)
        err_list.append(err*100)

        #comptime = float(words[COMPUTATION_TIME])
        comptime = float(words[COMPUTATION_TIME])
        comptime_list.append(comptime)

        totaltime = float(words[TOTAL_TIME])
        totaltime_list.append(totaltime)

        if_wall_list.append(float(words[IF_WALL_INDEX]))
        if_jerky_list.append(float(words[IF_JERKY_INDEX]))

    # when lose tracking happens
    lost_err_list = []
    for i in lost_list:
        lost_err_list.append(err_list[i])

    return [csr_list, icp_list, ir_list, vt_list, if_wall_list, if_jerky_list, err_list, comptime_list,
            totaltime_list], lost_list, lost_err_list, first_tracking_rate, tracking_rate

def process_energy(energy_file, num_frames):

    if not os.path.isfile(energy_file):
        return -1

    total_energy = 0.0

    f = open(energy_file)
    lines = f.readlines()
    f.close()

    for line in lines:
        words = line.split(',')
        total_energy += (float(words[2]) - idle_power)

    return total_energy/num_frames
    # return total_energy/len(lines)

def two_scales(ax1, time, data1, data2, c1, c2):
    ax2 = ax1.twinx()

    line1, = ax1.plot(time, data1, color=c1)
    ax1.set_xlabel('Frames')
    ax1.set_ylabel('Trajectory Error(m)')

    line2, = ax2.plot(time, data2, color=c2)
    ax2.set_ylabel('CSR(Frame Resolution)')
    ax2.set_ylim([0, 10])

    plt.legend((line1, line2), ('Trajectory Error','CSR'), loc='upper right')

    return ax1, ax2

def color_y_axis(ax, color):
    for t in ax.get_yticklabels():
        t.set_color(color)
    return None


if __name__ == "__main__":
    main()
