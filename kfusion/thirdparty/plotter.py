import sys
import math
import matplotlib.pyplot as plt

log = open(sys.argv[1],'r')
log_data = log.read()
log_lines = log_data.split("\n")
log_splited_lines = []
for i in log_lines:
    log_splited_lines += [i.split("\t")]

gt = open(sys.argv[2],'r')
gt_data = gt.read()
gt_lines = gt_data.split("\n")
gt_splited_lines = []
for i in gt_lines:
    gt_splited_lines += [i.split(" ")]

frame = []
delta_trans = []
delta_trans_x = []
delta_trans_y = []
delta_trans_z = []
delta_rot = []
delta_rot_x = []
delta_rot_y = []
delta_rot_z = []
tracked = []
trans_velocity = []
trans_velocity_x = []
trans_velocity_y = []
trans_velocity_z = []
rot_velocity_x = []
rot_velocity_y = []
rot_velocity_z = []
trans_accelaration = []
trans_accelaration_x = []
trans_accelaration_y = []
trans_accelaration_z = []
rot_accelaration_x = []
rot_accelaration_y = []
rot_accelaration_z = []

init_x = float(gt_splited_lines[0][1])
init_y = float(gt_splited_lines[0][2])
init_z = float(gt_splited_lines[0][3])


# print(gt_splited_lines)
# print(len(gt_splited_lines))
for i in range(len(gt_splited_lines)-1):
    frame += [float(log_splited_lines[i+1][0])]
    delta_tx = float(log_splited_lines[i+1][9]) - (float(gt_splited_lines[i][1]) - init_x)
    delta_ty = float(log_splited_lines[i+1][10]) + (float(gt_splited_lines[i][2]) - init_y)
    delta_tz = float(log_splited_lines[i+1][11]) - (float(gt_splited_lines[i][3]) - init_z)
    delta_trans += [math.sqrt(delta_tx**2 + delta_ty**2 +delta_tz**2)]
    delta_trans_x += [delta_tx]
    delta_trans_y += [delta_ty]
    delta_trans_z += [delta_tz]
    qx = float(gt_splited_lines[i][4])
    qy = float(gt_splited_lines[i][5])
    qz = float(gt_splited_lines[i][6])
    qw = float(gt_splited_lines[i][7])
    rx = -1*math.asin(2 * (qw*qx - qy*qz))
    ry = math.asin(2 * (qw*qy - qz*qx))
    rz = -1*math.asin(2 * (qw*qz - qx*qy))
    logrx = float(log_splited_lines[i+1][12])
    logry = float(log_splited_lines[i+1][13])
    logrz = float(log_splited_lines[i+1][14])
    if logrx > math.pi/2:
        logrx = math.pi - logrx
    if logrx < -1*math.pi/2:
        logrx = -logrx - math.pi
    if logry > math.pi/2:
        logry = math.pi - logry
    if logry < -1*math.pi/2:
        logry = -logry - math.pi
    if logrz > math.pi/2:
        logrz = math.pi - logrz
    if logrz < -1*math.pi/2:
        logrz = -logrz - math.pi
    delta_rx = logrx - rx
    delta_ry = logry - ry
    delta_rz = logrz - rz
    delta_rot_x += [delta_rx]
    delta_rot_y += [delta_ry]
    delta_rot_z += [delta_rz]
    delta_rot += [math.sqrt(delta_rx**2 + delta_ry**2 + delta_rz**2)]
    trans_velocity_x += [float(log_splited_lines[i+1][15])]
    trans_velocity_y += [float(log_splited_lines[i+1][16])]
    trans_velocity_z += [float(log_splited_lines[i+1][17])]
    trans_velocity += [math.sqrt(float(log_splited_lines[i+1][15])**2 +
                                 float(log_splited_lines[i+1][16])**2 +
                                 float(log_splited_lines[i+1][17])**2)]
    trans_accelaration_x += [float(log_splited_lines[i+1][18])]
    trans_accelaration_y += [float(log_splited_lines[i+1][19])]
    trans_accelaration_z += [float(log_splited_lines[i+1][20])]
    trans_accelaration += [math.sqrt(float(log_splited_lines[i+1][18])**2 +
                                     float(log_splited_lines[i+1][19])**2 +
                                     float(log_splited_lines[i+1][20])**2)]
    rot_velocity_x += [float(log_splited_lines[i+1][21])]
    rot_velocity_y += [float(log_splited_lines[i+1][22])]
    rot_velocity_z += [float(log_splited_lines[i+1][23])]
    rot_accelaration_x += [float(log_splited_lines[i+1][24])]
    rot_accelaration_y += [float(log_splited_lines[i+1][25])]
    rot_accelaration_z += [float(log_splited_lines[i+1][26])]
    tracked += [int(log_splited_lines[i+1][27])]

plt.subplot(3,3,1)
plt.title("X axis angle error")
plt.scatter(frame, delta_rot_x, c=tracked, marker=".")
plt.subplot(3,3,4)
plt.title("X axis rotational velocity")
plt.scatter(frame, rot_velocity_x, c=tracked, marker=".")
plt.subplot(3,3,7)
plt.title("X axis rotational acceleration")
plt.scatter(frame, rot_accelaration_x, c=tracked, marker=".")
plt.subplot(3,3,2)
plt.title("Y axis angle error")
plt.scatter(frame, delta_rot_y, c=tracked, marker=".")
plt.subplot(3,3,5)
plt.title("Y axis rotational velocity")
plt.scatter(frame, rot_velocity_y, c=tracked, marker=".")
plt.subplot(3,3,8)
plt.title("Y axis rotational acceleration")
plt.scatter(frame, rot_accelaration_y, c=tracked, marker=".")
plt.subplot(3,3,3)
plt.title("Z axis angle error")
plt.scatter(frame, delta_rot_z, c=tracked, marker= ".")
plt.subplot(3,3,6)
plt.title("Z axis rotational velocity")
plt.scatter(frame, rot_velocity_z, c=tracked, marker=".")
plt.subplot(3,3,9)
plt.title("Z axis rotational acceleration")
plt.scatter(frame, rot_accelaration_z, c=tracked, marker=".")
plt.subplots_adjust(hspace=0.3)
plt.show()

plt.subplot(3,3,1)
plt.title("X axis translation error")
plt.scatter(frame, delta_trans_x, c=tracked, marker=".")
plt.subplot(3,3,4)
plt.title("X axis translation velocity")
plt.scatter(frame, trans_velocity_x, c=tracked, marker=".")
plt.subplot(3,3,7)
plt.title("X axis translation acceleration")
plt.scatter(frame, trans_accelaration_x, c=tracked, marker=".")
plt.subplot(3,3,2)
plt.title("Y axis translation error")
plt.scatter(frame, delta_trans_y, c=tracked, marker=".")
plt.subplot(3,3,5)
plt.title("Y axis translation velocity")
plt.scatter(frame, trans_velocity_y, c=tracked, marker=".")
plt.subplot(3,3,8)
plt.title("Y axis translation acceleration")
plt.scatter(frame, trans_accelaration_y, c=tracked, marker=".")
plt.subplot(3,3,3)
plt.title("Z axis translation error")
plt.scatter(frame, delta_trans_z, c=tracked, marker= ".")
plt.subplot(3,3,6)
plt.title("Z axis translation velocity")
plt.scatter(frame, trans_velocity_z, c=tracked, marker=".")
plt.subplot(3,3,9)
plt.title("Z axis translation acceleration")
plt.scatter(frame, trans_accelaration_z, c=tracked, marker=".")
plt.subplots_adjust(hspace=0.3)
plt.show()

plt.subplot(3,1,1)
plt.title("Absolute translation error")
plt.scatter(frame, delta_trans, c=tracked, marker=".")
plt.subplot(3,1,2)
plt.title("Absolute translation velocity")
plt.scatter(frame, trans_velocity, c=tracked, marker=".")
plt.subplot(3,1,3)
plt.title("Absolute translation acceleration")
plt.scatter(frame, trans_accelaration, c=tracked, marker=".")
plt.subplots_adjust(hspace=0.3)
plt.show()
