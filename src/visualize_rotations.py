# -*- coding: utf-8 -*-
"""
旋转函数可视化验证
通过3D图形直观展示旋转矩阵的效果
"""
import numpy as np
import matplotlib.pyplot as plt
import angle_function as af

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def plot_coordinate_frame(ax, R, origin=np.array([0, 0, 0]), scale=1.0, label_prefix=''):
    """绘制坐标系"""
    colors = ['r', 'g', 'b']
    labels = ['X', 'Y', 'Z']

    for i in range(3):
        axis = R[:, i] * scale
        ax.quiver(origin[0], origin[1], origin[2],
                  axis[0], axis[1], axis[2],
                  color=colors[i], arrow_length_ratio=0.1, linewidth=2)
        ax.text(origin[0] + axis[0] * 1.1,
                origin[1] + axis[1] * 1.1,
                origin[2] + axis[2] * 1.1,
                f'{label_prefix}{labels[i]}', color=colors[i], fontsize=10)


def visualize_basic_rotations():
    """可视化基本旋转矩阵 rotx, roty, rotz"""
    fig = plt.figure(figsize=(15, 6))

    angles = [0, np.pi/4, np.pi/2]
    angle_labels = ['0°', '45°', '90°']

    # rotx 可视化
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title('rotx - 绕X轴旋转', fontsize=14, pad=15)
    plot_coordinate_frame(ax1, np.eye(3), label_prefix='原')
    for angle, label in zip(angles[1:], angle_labels[1:]):
        R = af.rotx(angle)
        plot_coordinate_frame(ax1, R, scale=0.8)
    ax1.set_xlim([-1.5, 1.5])
    ax1.set_ylim([-1.5, 1.5])
    ax1.set_zlim([-1.5, 1.5])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # roty 可视化
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title('roty - 绕Y轴旋转', fontsize=14, pad=15)
    plot_coordinate_frame(ax2, np.eye(3), label_prefix='原')
    for angle, label in zip(angles[1:], angle_labels[1:]):
        R = af.roty(angle)
        plot_coordinate_frame(ax2, R, scale=0.8)
    ax2.set_xlim([-1.5, 1.5])
    ax2.set_ylim([-1.5, 1.5])
    ax2.set_zlim([-1.5, 1.5])
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')


    # rotz 可视化
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_title('rotz - 绕Z轴旋转', fontsize=14, pad=15)
    plot_coordinate_frame(ax3, np.eye(3), label_prefix='原')
    for angle, label in zip(angles[1:], angle_labels[1:]):
        R = af.rotz(angle)
        plot_coordinate_frame(ax3, R, scale=0.8)
    ax3.set_xlim([-1.5, 1.5])
    ax3.set_ylim([-1.5, 1.5])
    ax3.set_zlim([-1.5, 1.5])
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('../images/basic_rotations.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_euler_angles():
    """可视化欧拉角转换"""
    fig = plt.figure(figsize=(12, 6))

    euler = [np.pi/6, np.pi/4, np.pi/3]  # roll, pitch, yaw

    # 内旋 rxyz
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title(f'内旋 rxyz\nroll={np.degrees(euler[0]):.0f}°, pitch={np.degrees(euler[1]):.0f}°, yaw={np.degrees(euler[2]):.0f}°',
                  fontsize=12, pad=15)
    plot_coordinate_frame(ax1, np.eye(3), label_prefix='原')
    R_rxyz = af.euler_to_rot(euler, "rxyz")
    plot_coordinate_frame(ax1, R_rxyz, scale=0.9)
    ax1.set_xlim([-1.5, 1.5])
    ax1.set_ylim([-1.5, 1.5])
    ax1.set_zlim([-1.5, 1.5])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # 外旋 sxyz
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title(f'外旋 sxyz\nroll={np.degrees(euler[0]):.0f}°, pitch={np.degrees(euler[1]):.0f}°, yaw={np.degrees(euler[2]):.0f}°',
                  fontsize=12, pad=15)
    plot_coordinate_frame(ax2, np.eye(3), label_prefix='原')
    R_sxyz = af.euler_to_rot(euler, "sxyz")
    plot_coordinate_frame(ax2, R_sxyz, scale=0.9)
    ax2.set_xlim([-1.5, 1.5])
    ax2.set_ylim([-1.5, 1.5])
    ax2.set_zlim([-1.5, 1.5])
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('../images/euler_angles.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_rodrigues():
    """可视化罗德里格斯旋转公式"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('罗德里格斯旋转 - 绕任意轴旋转', fontsize=14, pad=15)

    # 原始坐标系
    plot_coordinate_frame(ax, np.eye(3), label_prefix='原')

    # 绕对角线轴旋转
    axis = np.array([1, 1, 1]) / np.sqrt(3)
    angles = [np.pi/6, np.pi/3, np.pi/2]

    # 绘制旋转轴
    ax.quiver(0, 0, 0, axis[0]*1.5, axis[1]*1.5, axis[2]*1.5,
              color='purple', arrow_length_ratio=0.1, linewidth=3, label='旋转轴')

    for angle in angles:
        R = af.rotk(axis, angle)
        plot_coordinate_frame(ax, R, scale=0.7)

    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('../images/rodrigues_rotation.png', dpi=150, bbox_inches='tight')
    plt.show()



def visualize_quaternion():
    """可视化四元数旋转"""
    fig = plt.figure(figsize=(12, 6))

    # 绕X轴旋转的四元数
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('四元数旋转 - 绕X轴', fontsize=14, pad=15)
    plot_coordinate_frame(ax1, np.eye(3), label_prefix='原')

    for angle in [np.pi/4, np.pi/2, np.pi*3/4]:
        quat = np.array([np.cos(angle/2), np.sin(angle/2), 0, 0])
        R = af.quat_to_rot(quat)
        plot_coordinate_frame(ax1, R, scale=0.8)

    ax1.set_xlim([-1.5, 1.5])
    ax1.set_ylim([-1.5, 1.5])
    ax1.set_zlim([-1.5, 1.5])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # 任意四元数
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('四元数旋转 - 任意轴', fontsize=14, pad=15)
    plot_coordinate_frame(ax2, np.eye(3), label_prefix='原')

    # 绕(1,1,1)轴旋转
    axis = np.array([1, 1, 1]) / np.sqrt(3)
    for angle in [np.pi/4, np.pi/2]:
        quat = np.array([np.cos(angle/2),
                         axis[0]*np.sin(angle/2),
                         axis[1]*np.sin(angle/2),
                         axis[2]*np.sin(angle/2)])
        R = af.quat_to_rot(quat)
        plot_coordinate_frame(ax2, R, scale=0.8)

    ax2.set_xlim([-1.5, 1.5])
    ax2.set_ylim([-1.5, 1.5])
    ax2.set_zlim([-1.5, 1.5])
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('../images/quaternion_rotation.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_homogeneous_transform():
    """可视化齐次变换矩阵"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('齐次变换 - 旋转 + 平移', fontsize=14, pad=15)

    # 原始坐标系
    plot_coordinate_frame(ax, np.eye(3), origin=np.array([0, 0, 0]), label_prefix='原')

    # 变换后的坐标系
    R = af.euler_to_rot([np.pi/6, np.pi/4, 0], "rxyz")
    p = np.array([2, 1, 0.5])
    trans = af.rp_to_trans(R, p)

    R_trans, p_trans = af.trans_to_rp(trans)
    plot_coordinate_frame(ax, R_trans, origin=p_trans, label_prefix='变')

    # 绘制平移向量
    ax.quiver(0, 0, 0, p[0], p[1], p[2],
              color='purple', arrow_length_ratio=0.05, linewidth=2,
              linestyle='--', label=f'平移 ({p[0]}, {p[1]}, {p[2]})')

    ax.set_xlim([-1, 4])
    ax.set_ylim([-1, 3])
    ax.set_zlim([-1, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('../images/homogeneous_transform.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_transform_inverse():
    """可视化变换矩阵求逆"""
    fig = plt.figure(figsize=(12, 6))

    R = af.rotz(np.pi/3)
    p = np.array([2, 1, 0])
    trans = af.rp_to_trans(R, p)
    trans_inv = af.trans_inv(trans)

    # 原始变换
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('原始变换 T', fontsize=14, pad=15)
    plot_coordinate_frame(ax1, np.eye(3), origin=np.array([0, 0, 0]), label_prefix='世界')
    R1, p1 = af.trans_to_rp(trans)
    plot_coordinate_frame(ax1, R1, origin=p1, label_prefix='T')
    ax1.quiver(0, 0, 0, p1[0], p1[1], p1[2], color='purple',
               arrow_length_ratio=0.05, linewidth=2, linestyle='--')
    ax1.set_xlim([-1, 4])
    ax1.set_ylim([-1, 3])
    ax1.set_zlim([-1, 2])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # 逆变换
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('逆变换 T_inv', fontsize=14, pad=15)
    plot_coordinate_frame(ax2, np.eye(3), origin=np.array([0, 0, 0]), label_prefix='世界')
    R2, p2 = af.trans_to_rp(trans_inv)
    plot_coordinate_frame(ax2, R2, origin=p2, label_prefix='Tinv')
    ax2.quiver(0, 0, 0, p2[0], p2[1], p2[2], color='purple',
               arrow_length_ratio=0.05, linewidth=2, linestyle='--')
    ax2.set_xlim([-3, 2])
    ax2.set_ylim([-2, 2])
    ax2.set_zlim([-1, 2])
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('../images/transform_inverse.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    print("=" * 50)
    print("旋转函数可视化验证")
    print("=" * 50)

    print("\n1. 基本旋转矩阵 (rotx, roty, rotz)")
    visualize_basic_rotations()

    print("\n2. 欧拉角转换 (内旋 vs 外旋)")
    visualize_euler_angles()

    print("\n3. 罗德里格斯旋转公式")
    visualize_rodrigues()

    print("\n4. 四元数旋转")
    visualize_quaternion()

    print("\n5. 齐次变换矩阵")
    visualize_homogeneous_transform()

    print("\n6. 变换矩阵求逆")
    visualize_transform_inverse()

    print("\n可视化完成！图片已保存。")
