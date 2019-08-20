import sys
import math
import matplotlib.pyplot as plt
import numpy as np

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
kalman_tx = []
kalman_ty = []
kalman_tz = []
kalman_rx = []
kalman_ry = []
kalman_rz = []
log_trans_x = []
log_trans_y = []
log_trans_z = []
gt_trans_x = []
gt_trans_y = []
gt_trans_z = []

init_x = float(gt_splited_lines[0][1])
init_y = float(gt_splited_lines[0][2])
init_z = float(gt_splited_lines[0][3])

k_q = np.matrix([[0.25*(1/30)**4, 0.5*(1/30)**3],
                [0.5*(1/30)**3, 1/30]])
k_q = k_q * float(sys.argv[3]) # acceleration standard deviation squared

k_r = float(sys.argv[4]) # mesurement standard deviation squared

k_a = np.matrix([[1.0, 1.0/30],
                   [0.0, 1.0]])

k_h = np.matrix([1.0 ,0.0])


k_state_tx = np.matrix([[0.0], 
                        [0.0]])
k_p_tx = np.matrix([[0.0, 0.0],
                   [0.0, 0.0]])
k_state_ty = np.matrix([[0.0], 
                        [0.0]])
k_p_ty = np.matrix([[0.0, 0.0],
                   [0.0, 0.0]])
k_state_tz = np.matrix([[0.0], 
                        [0.0]])
k_p_tz = np.matrix([[0.0, 0.0],
                   [0.0, 0.0]])

k_state_rx = np.matrix([[0.0], 
                        [0.0]])
k_p_rx = np.matrix([[0.0, 0.0],
                   [0.0, 0.0]])
k_state_ry = np.matrix([[0.0], 
                        [0.0]])
k_p_ry = np.matrix([[0.0, 0.0],
                   [0.0, 0.0]])
k_state_rz = np.matrix([[0.0], 
                        [0.0]])
k_p_rz = np.matrix([[0.0, 0.0],
                   [0.0, 0.0]])

# print(gt_splited_lines)
# print(len(gt_splited_lines))
for i in range(len(gt_splited_lines)-1):
    frame += [float(log_splited_lines[i+1][0])]
    log_trans_x += [float(log_splited_lines[i+1][9])]
    gt_trans_x += [float(gt_splited_lines[i][1]) - init_x]
    delta_tx = float(log_splited_lines[i+1][9]) - (float(gt_splited_lines[i][1]) - init_x)

    log_trans_y += [float(log_splited_lines[i+1][10])]
    gt_trans_y += [init_y - float(gt_splited_lines[i][2])]
    delta_ty = float(log_splited_lines[i+1][10]) + (float(gt_splited_lines[i][2]) - init_y)

    log_trans_z += [float(log_splited_lines[i+1][11])]
    gt_trans_z += [float(gt_splited_lines[i][3]) - init_z]
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
    k_pred_state_tx = k_a * k_state_tx
    k_pred_p_tx = k_a * k_p_tx * k_a.T + k_q
    k_k = (k_pred_p_tx * k_h.T)/(k_h * k_pred_p_tx * k_h.T + k_r)
    k_y_tx = float(log_splited_lines[i+1][9]) - k_h * k_pred_state_tx
    k_state_tx = k_pred_state_tx + k_k * k_y_tx
    k_p_tx = (np.matrix(np.identity(2)) - k_k * k_h) * k_pred_p_tx
    kalman_tx += [k_pred_state_tx[0] - (float(gt_splited_lines[i][1]) - init_x)
]

    k_pred_state_ty = k_a * k_state_ty
    k_pred_p_ty = k_a * k_p_ty * k_a.T + k_q
    k_k = (k_pred_p_ty * k_h.T)/(k_h * k_pred_p_ty * k_h.T + k_r)
    k_y_ty = float(log_splited_lines[i+1][10]) - k_h * k_pred_state_ty
    k_state_ty = k_pred_state_ty + k_k * k_y_ty
    k_p_ty = (np.matrix(np.identity(2)) - k_k * k_h) * k_pred_p_ty
    kalman_ty += [k_pred_state_ty[0] + (float(gt_splited_lines[i][1]) - init_x)
]

    k_pred_state_tz = k_a * k_state_tz
    k_pred_p_tz = k_a * k_p_tz * k_a.T + k_q
    k_k = (k_pred_p_tz * k_h.T)/(k_h * k_pred_p_tz * k_h.T + k_r)
    k_y_tz = float(log_splited_lines[i+1][11]) - k_h * k_pred_state_tz
    k_state_tz = k_pred_state_tz + k_k * k_y_tz
    k_p_tz = (np.matrix(np.identity(2)) - k_k * k_h) * k_pred_p_tz
    kalman_tz += [k_pred_state_tz[0] - (float(gt_splited_lines[i][1]) - init_x)
]


    k_pred_state_rx = k_a * k_state_rx
    k_pred_p_rx = k_a * k_p_rx * k_a.T + k_q
    k_k = (k_pred_p_rx * k_h.T)/(k_h * k_pred_p_rx * k_h.T + k_r)
    k_y_rx = logrx - k_h * k_pred_state_rx
    k_state_rx = k_pred_state_rx + k_k * k_y_rx
    k_p_rx = (np.matrix(np.identity(2)) - k_k * k_h) * k_pred_p_rx
    kalman_rx += [k_state_rx[0] - rx]

    k_pred_state_ry = k_a * k_state_ry
    k_pred_p_ry = k_a * k_p_ry * k_a.T + k_q
    k_k = (k_pred_p_ry * k_h.T)/(k_h * k_pred_p_ry * k_h.T + k_r)
    k_y_ry = logry - k_h * k_pred_state_ry
    k_state_ry = k_pred_state_ry + k_k * k_y_ry
    k_p_ry = (np.matrix(np.identity(2)) - k_k * k_h) * k_pred_p_ry
    kalman_ry += [k_state_ry[0] - ry]

    k_pred_state_rz = k_a * k_state_rz
    k_pred_p_rz = k_a * k_p_rz * k_a.T + k_q
    k_k = (k_pred_p_rz * k_h.T)/(k_h * k_pred_p_rz * k_h.T + k_r)
    k_y_rz = logrz - k_h * k_pred_state_rz
    k_state_rz = k_pred_state_rz + k_k * k_y_rz
    k_p_rz = (np.matrix(np.identity(2)) - k_k * k_h) * k_pred_p_rz
    kalman_rz += [k_state_rz[0] - rz]


print(np.std(rot_accelaration_x)**2)
print(np.std(rot_accelaration_y)**2)
print(np.std(rot_accelaration_z)**2)
print(np.std(delta_rot_x)**2)
print(np.std(delta_rot_y)**2)
print(np.std(delta_rot_z)**2)

print(np.std(trans_accelaration_x)**2)
print(np.std(trans_accelaration_y)**2)
print(np.std(trans_accelaration_z)**2)
print(np.std(delta_trans_x)**2)
print(np.std(delta_trans_y)**2)
print(np.std(delta_trans_z)**2)

plt.subplot(3,3,1)
plt.title("X axis angle error")
plt.scatter(frame, delta_rot_x, c=tracked, marker=".")
plt.scatter(frame, kalman_rx, c=tracked, marker="x")
plt.subplot(3,3,4)
plt.title("X axis rotational velocity")
plt.scatter(frame, rot_velocity_x, c=tracked, marker=".")
plt.subplot(3,3,7)
plt.title("X axis rotational acceleration")
plt.scatter(frame, rot_accelaration_x, c=tracked, marker=".")
plt.subplot(3,3,2)
plt.title("Y axis angle error")
plt.scatter(frame, delta_rot_y, c=tracked, marker=".")
plt.scatter(frame, kalman_ry, c=tracked, marker="x")
plt.subplot(3,3,5)
plt.title("Y axis rotational velocity")
plt.scatter(frame, rot_velocity_y, c=tracked, marker=".")
plt.subplot(3,3,8)
plt.title("Y axis rotational acceleration")
plt.scatter(frame, rot_accelaration_y, c=tracked, marker=".")
plt.subplot(3,3,3)
plt.title("Z axis angle error")
plt.scatter(frame, delta_rot_z, c=tracked, marker= ".")
plt.scatter(frame, kalman_rz, c=tracked, marker="x")
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
plt.scatter(frame, kalman_tx, c=tracked, marker="x")
plt.subplot(3,3,4)
plt.title("X axis translation velocity")
plt.scatter(frame, trans_velocity_x, c=tracked, marker=".")
plt.subplot(3,3,7)
plt.title("X axis translation acceleration")
plt.scatter(frame, trans_accelaration_x, c=tracked, marker=".")
plt.subplot(3,3,2)
plt.title("Y axis translation error")
plt.scatter(frame, delta_trans_y, c=tracked, marker=".")
plt.scatter(frame, kalman_ty, c=tracked, marker="x")
plt.subplot(3,3,5)
plt.title("Y axis translation velocity")
plt.scatter(frame, trans_velocity_y, c=tracked, marker=".")
plt.subplot(3,3,8)
plt.title("Y axis translation acceleration")
plt.scatter(frame, trans_accelaration_y, c=tracked, marker=".")
plt.subplot(3,3,3)
plt.title("Z axis translation error")
plt.scatter(frame, delta_trans_z, c=tracked, marker=".")
plt.scatter(frame, kalman_tz, c=tracked, marker="x")
plt.subplot(3,3,6)
plt.title("Z axis translation velocity")
plt.scatter(frame, trans_velocity_z, c=tracked, marker=".")
plt.subplot(3,3,9)
plt.title("Z axis translation acceleration")
plt.scatter(frame, trans_accelaration_z, c=tracked, marker=".")
plt.subplots_adjust(hspace=0.3)
plt.show()
"""
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
"""
