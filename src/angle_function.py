# -*- coding: utf-8 -*-
import numpy as np

"""
旋转方式
rxyz: 内旋，绕自身局部坐标系旋转，如机器人TCP坐标系
sxyz: 外旋，绕全局固定坐标系旋转，如世界坐标系

内旋旋转矩阵(rxyz)
R = Rx(roll) @ Ry(pitch) @ Rz(yaw)
外旋旋转矩阵(sxyz)
R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
"""

def rotx(angle):
    """
    计算绕 X 轴旋转 angle 角度的旋转矩阵
    :param angle: 旋转角度
    :return: 3x3 旋转矩阵
    """
    mat = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(angle), -np.sin(angle)],
        [0.0, np.sin(angle), np.cos(angle)]
    ])

    return mat

def roty(angle):
    """
    计算绕 Y 轴旋转 angle 角度的旋转矩阵
    :param angle: 旋转角度
    :return: 3x3 旋转矩阵
    """
    mat = np.array([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)]
    ])

    return mat

def rotz(angle):
    """
    计算绕 Z 轴旋转 angle 角度的旋转矩阵
    :param angle: 旋转角度
    :return: 3x3 旋转矩阵
    """
    mat = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0]
    ])

    return mat

def euler_to_rot(euler_angles, order="rxyz"):
    """
    欧拉角转旋转矩阵
    :param euler_angles: 欧拉角 [roll, pitch, yaw]
    :param order: 旋转方式, 默认为 rxyz
    :Return: 3x3 旋转矩阵
    """
    is_static = order[0] == "s" # 是否为外旋
    map_matrix = {}             # 存储三个轴的旋转函数
    map_matrix["x"] = rotx
    map_matrix["y"] = roty
    map_matrix["z"] = rotz
    res = np.eye(3)           # 创建单位矩阵
    for i in range(3):          # 按照给定的旋转顺序进行旋转
        axis = order[i + 1]
        mat = map_matrix[axis](euler_angles[i])

        if is_static:
            res = mat @ res     # 外旋，新旋转矩阵左乘当前矩阵
        else:
            res = res @ mat     # 内旋，新旋转矩阵右乘当前矩阵

    return res

def hat(axis):
    """
    将三维向量转换为反对称矩阵(hat 算子)
    :param axis: 旋转轴
    :return: 3x3 反对称矩阵
    """
    if len(axis) != 3:
        raise ValueError("axis must be a 3-element vector")
    mat = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0]
    ])

    return mat

def rotk(axis, angle):
    """
    通过罗德里格斯旋转公式计算绕任意轴旋转 angle 的旋转矩阵
    R = I + sin(angle)K + (1 - cos(angle))K^2
    :param axis: 旋转轴
    :param angle: 旋转角度
    :return: 3x3 旋转矩阵
    """
    hat_mat = hat(axis)
    mat = np.eye(3) + np.sin(angle) * hat_mat + (1 - np.cos(angle)) * (hat_mat @ hat_mat)

    return mat

def quat_to_rot(quat):
    """
    四元数转旋转矩阵
    :param quat: 四元数 [w, x, y, z]
    :return: 3x3 旋转矩阵
    """
    quat = quat / np.linalg.norm(quat)
    qw, qx, qy, qz = quat
    rot = np.array([
        [1 - 2 * qy ** 2 - 2 * qz ** 2,
         2 * qx * qy - 2 * qz * qw,
         2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw,
         1 - 2 * qx ** 2 - 2 * qz ** 2,
         2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw,
         2 * qy * qz + 2 * qx * qw,
         1 - 2 * qx ** 2 - 2 * qy ** 2]
    ])

    return rot

def rot_to_quat(rot):
    """
    旋转矩阵转四元数
    使用旋转矩阵的 trace 值来计算 w 分量
    使用旋转矩阵的反对称部分来计算 x, y, z 分量
    :param rot: 旋转矩阵
    :return: 归一化的四元数
    """
    qw = np.sqrt(1 + rot[0, 0] + rot[1, 1] + rot[2, 2]) / 2
    qx = (rot[2, 1] - rot[1, 2]) / (4 * qw)
    qy = (rot[0, 2] - rot[2, 0]) / (4 * qw)
    qz = (rot[1, 0] - rot[0, 1]) / (4 * qw)
    res = np.array([qw, qx, qy, qz])
    res = res / np.linalg.norm(res)

    return res

def rot_inv(R):
    """
    计算旋转矩阵的逆矩阵
    :param R: 旋转矩阵
    :return: 旋转矩阵的逆矩阵
    """
    return R.transpose()

def vec3_to_so3(vec3):
    """
    将三维向量转换为反对称矩阵
    :param vec3: 三维向量
    :return: 3x3 反对称矩阵
    """
    return hat(vec3)

def so3_to_vec3(so3mat):
    """
    将反对称矩阵转换为三维向量
    :param so3mat: 3x3 反对称矩阵
    :return: 三维向量
    """
    vec3 = np.array([so3mat[2, 1], so3mat[0, 2], so3mat[1, 0]])
    return vec3

def axis_angle(expc3):
    """
    将三维向量转换为轴角表示
    :param expc3: 三维向量
    :return: 轴向量和角度
    """
    norm = np.linalg.norm(expc3)    # 计算向量范数
    axis= expc3.copy() / norm         # 归一化向量作为旋转轴
    angle = norm

    return axis, angle

def matrix_to_exp3(so3mat):
    """
    将反对称矩阵转换为旋转矩阵
    :param so3mat: 3x3 反对称矩阵
    :return: 3x3 旋转矩阵
    """
    vec = so3_to_vec3(so3mat)
    angle = np.linalg.norm(vec)
    axis = vec / angle

    return rotk(axis, angle)

def matrix_log3(R):
    """
    计算旋转矩阵的对数矩阵
    :param R: 3x3 旋转矩阵
    :return: 3x3 对数矩阵
    """
    angle = np.arccos((np.trace(R) - 1) / 2)    # 计算旋转角度
    omega = 1.0 / (2 * np.sin(angle)) * (R - R.transpose())

    return angle * omega

def rp_to_trans(R, p):
    """
    将旋转矩阵和平移向量转换为齐次变换矩阵
    :param R: 3x3 旋转矩阵
    :param p: 3x1 平移向量
    :return: 4x4 齐次变换矩阵
    """
    trans = np.eye(4)
    trans[:3, :3] = R
    trans[:3, 3] = p

    return trans

def trans_to_rp(trans):
    """
    将齐次变换矩阵转换为旋转矩阵 + 平移向量
    :param trans: 4x4 齐次变换矩阵
    :return: 旋转矩阵和平移向量
    """
    R = trans[:3, :3]
    p = trans[:3, 3]

    return R, p

def trans_inv(trans):
    """
    计算齐次变换矩阵的逆矩阵
    :param trans: 4x4 齐次变换矩阵
    :return: 4x4 齐次变换矩阵的逆矩阵
    """
    trans_inv = np.eye(4)
    trans_inv[:3, :3] = trans[:3, :3].transpose()
    trans_inv[:3, 3] = -trans[:3, :3].transpose() @ trans[:3, 3]

    return trans_inv

def vec6_to_se3(vec6):
    """
    将六维向量转换为 se3 矩阵
    :param vec6: 六维向量
    :return: 4x4 se3 矩阵
    """
    se3_mat = np.zeros((4, 4))
    se3_mat[:3, :3] = hat(vec6[:3])
    se3_mat[:3, 3] = vec6[3:]

    return se3_mat

def se3_to_vec6(se3_mat):
    """
    将 se3 矩阵转换为六维向量
    :param se3_mat: 4x4 se3 矩阵
    :return: 六维向量
    """
    omega = so3_to_vec3(se3_mat[:3, :3])
    vbec = se3_mat[:3, 3]
    vec6 = np.zeros(6)
    vec6[:3] = omega
    vec6[3:] = vbec

    return vec6

def adjoint_mat(trans):
    """
    计算变换矩阵的 6x6 伴随矩阵，用于在不同坐标系之间转换速度和力
    :param trans: 旋转矩阵
    :return: 6x6 伴随矩阵
    """
    R, p = trans_to_rp(trans)
    adjoint_mat = np.zeros((6, 6))
    adjoint_mat[:3, :3] = R
    adjoint_mat[:3, 3:] = hat(p) @ R
    adjoint_mat[3:, 3:] = R

    return adjoint_mat

def screw_to_axis(q, s, h):
    """
    将螺旋运动转换为螺旋轴
    :param q: 螺旋轴上的一点
    :param s: 螺旋轴的方向向量
    :param h: 螺旋的升角(螺距)
    :return: 螺旋轴
    """
    vec6 = np.zeros(6)
    omega = s / np.linalg.norm(s)
    vec6[:3] = omega
    vec6[3:] = -np.cross(omega, q) + h * omega

    return vec6

def axis_angle6(expc6):
    """
    将旋转向量转换为旋转轴和旋转角度
    :param expc6: 六维旋转向量，前三个元素表示旋转向量
    :return: 旋转轴 S 和旋转角度 angle
    """
    angle = np.linalg.norm(expc6[:3])
    S = expc6 / angle

    return S, angle

def matrix_to_exp6(se3_mat):
    """
    将 se3 矩阵转换为六维旋转向量，目的是将齐次变换矩阵转换为旋转向量
    :param se3_mat: 4x4 se3 变换矩阵
    :return: 六维旋转向量
    """
    vec3 = so3_to_vec3(se3_mat[:3, :3])
    print(vec3)
    angle = np.linalg.norm(vec3)
    print(angle)
    v = se3_mat[:3, 3] / angle
    omega = vec3 / angle
    trans = np.eye(4)
    trans[:3, :3] = rotk(omega, angle)
    trans[:3, 3] = (
        np.eye(3) * angle + hat(omega) * (1 - np.cos(angle)) + hat(omega) @ hat(omega) * (angle - np.sin(angle))
    ) @ v

    return trans

def matrix_log6(trans):
    """
    计算齐次变换矩阵的对数矩阵
    :param trans: 齐次变换矩阵
    :return: 4x4 对数变换矩阵
    """
    R = trans[:3, :3]
    vec = so3_to_vec3(matrix_log3(R))
    angle = np.linalg.norm(vec)
    omega = vec / angle
    p = trans[:3, 3]
    v = (
        (1.0 / angle) * (np.eye(3) - 0.5 * hat(omega) + (1.0 / angle - 1.0 / 2.0 * 1.0 / np.tan(angle / 2.0)) * hat(omega) @ hat(omega))
    ) @ p

    se3_mat = np.zeros((4, 4))
    se3_mat[:3, :3] = hat(omega) * angle
    se3_mat[:3, 3] = v * angle

    return se3_mat
