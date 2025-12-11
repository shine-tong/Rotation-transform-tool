# -*- coding: utf-8 -*-
"""
angle_function.py 单元测试
测试所有旋转矩阵、四元数、欧拉角相关函数的正确性
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
import angle_function as af


class TestRotationMatrices:
    """测试基本旋转矩阵函数"""
    
    def test_rotx_zero_angle(self):
        """绕X轴旋转0度应返回单位矩阵"""
        result = af.rotx(0)
        expected = np.eye(3)
        assert_array_almost_equal(result, expected)
    
    def test_rotx_90_degrees(self):
        """绕X轴旋转90度"""
        angle = np.pi / 2
        result = af.rotx(angle)
        expected = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        assert_array_almost_equal(result, expected, decimal=10)
    
    def test_rotx_is_orthogonal(self):
        """rotx应返回正交矩阵 (R @ R.T = I)"""
        angle = np.pi / 4
        R = af.rotx(angle)
        assert_array_almost_equal(R @ R.T, np.eye(3))
        assert_allclose(np.linalg.det(R), 1.0)
    
    def test_roty_zero_angle(self):
        """绕Y轴旋转0度应返回单位矩阵"""
        result = af.roty(0)
        expected = np.eye(3)
        assert_array_almost_equal(result, expected)
    
    def test_roty_90_degrees(self):
        """绕Y轴旋转90度"""
        angle = np.pi / 2
        result = af.roty(angle)
        expected = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]
        ])
        assert_array_almost_equal(result, expected, decimal=10)
    
    def test_roty_is_orthogonal(self):
        """roty应返回正交矩阵"""
        angle = np.pi / 3
        R = af.roty(angle)
        assert_array_almost_equal(R @ R.T, np.eye(3))
        assert_allclose(np.linalg.det(R), 1.0)

    
    def test_rotz_zero_angle(self):
        """绕Z轴旋转0度应返回单位矩阵"""
        result = af.rotz(0)
        expected = np.eye(3)
        assert_array_almost_equal(result, expected)
    
    def test_rotz_90_degrees(self):
        """绕Z轴旋转90度"""
        angle = np.pi / 2
        result = af.rotz(angle)
        expected = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        assert_array_almost_equal(result, expected, decimal=10)
    
    def test_rotz_is_orthogonal(self):
        """rotz应返回正交矩阵"""
        angle = np.pi / 6
        R = af.rotz(angle)
        assert_array_almost_equal(R @ R.T, np.eye(3))
        assert_allclose(np.linalg.det(R), 1.0)


class TestEulerToRot:
    """测试欧拉角转旋转矩阵"""
    
    def test_euler_zero_angles(self):
        """零欧拉角应返回单位矩阵"""
        result = af.euler_to_rot([0, 0, 0], "rxyz")
        expected = np.eye(3)
        assert_array_almost_equal(result, expected)
    
    def test_euler_rxyz_single_axis(self):
        """内旋单轴旋转"""
        angle = np.pi / 4
        result = af.euler_to_rot([angle, 0, 0], "rxyz")
        expected = af.rotx(angle)
        assert_array_almost_equal(result, expected)
    
    def test_euler_sxyz_single_axis(self):
        """外旋单轴旋转"""
        angle = np.pi / 4
        result = af.euler_to_rot([angle, 0, 0], "sxyz")
        expected = af.rotx(angle)
        assert_array_almost_equal(result, expected)
    
    def test_euler_result_is_orthogonal(self):
        """欧拉角转换结果应为正交矩阵"""
        euler = [np.pi/6, np.pi/4, np.pi/3]
        R = af.euler_to_rot(euler, "rxyz")
        assert_array_almost_equal(R @ R.T, np.eye(3))
        assert_allclose(np.linalg.det(R), 1.0)


class TestHatOperator:
    """测试hat算子（反对称矩阵）"""
    
    def test_hat_basic(self):
        """基本hat运算"""
        axis = np.array([1, 2, 3])
        result = af.hat(axis)
        expected = np.array([
            [0, -3, 2],
            [3, 0, -1],
            [-2, 1, 0]
        ])
        assert_array_almost_equal(result, expected)
    
    def test_hat_is_skew_symmetric(self):
        """hat结果应为反对称矩阵"""
        axis = np.array([1.5, -2.3, 0.7])
        result = af.hat(axis)
        assert_array_almost_equal(result, -result.T)
    
    def test_hat_invalid_input(self):
        """非3维向量应抛出异常"""
        with pytest.raises(ValueError):
            af.hat(np.array([1, 2]))


class TestRotk:
    """测试罗德里格斯旋转公式"""
    
    def test_rotk_x_axis(self):
        """绕X轴旋转应与rotx一致"""
        axis = np.array([1, 0, 0])
        angle = np.pi / 4
        result = af.rotk(axis, angle)
        expected = af.rotx(angle)
        assert_array_almost_equal(result, expected)
    
    def test_rotk_y_axis(self):
        """绕Y轴旋转应与roty一致"""
        axis = np.array([0, 1, 0])
        angle = np.pi / 3
        result = af.rotk(axis, angle)
        expected = af.roty(angle)
        assert_array_almost_equal(result, expected)
    
    def test_rotk_z_axis(self):
        """绕Z轴旋转应与rotz一致"""
        axis = np.array([0, 0, 1])
        angle = np.pi / 6
        result = af.rotk(axis, angle)
        expected = af.rotz(angle)
        assert_array_almost_equal(result, expected)
    
    def test_rotk_is_orthogonal(self):
        """rotk结果应为正交矩阵"""
        axis = np.array([1, 1, 1]) / np.sqrt(3)
        angle = np.pi / 5
        R = af.rotk(axis, angle)
        assert_array_almost_equal(R @ R.T, np.eye(3))
        assert_allclose(np.linalg.det(R), 1.0)



class TestQuaternion:
    """测试四元数相关函数"""
    
    def test_quat_to_rot_identity(self):
        """单位四元数应返回单位矩阵"""
        quat = np.array([1, 0, 0, 0])
        result = af.quat_to_rot(quat)
        expected = np.eye(3)
        assert_array_almost_equal(result, expected)
    
    def test_quat_to_rot_90_x(self):
        """绕X轴旋转90度的四元数"""
        angle = np.pi / 2
        quat = np.array([np.cos(angle/2), np.sin(angle/2), 0, 0])
        result = af.quat_to_rot(quat)
        expected = af.rotx(angle)
        assert_array_almost_equal(result, expected)
    
    def test_quat_to_rot_is_orthogonal(self):
        """四元数转换结果应为正交矩阵"""
        quat = np.array([0.5, 0.5, 0.5, 0.5])
        R = af.quat_to_rot(quat)
        assert_array_almost_equal(R @ R.T, np.eye(3))
        assert_allclose(np.linalg.det(R), 1.0)
    
    def test_rot_to_quat_identity(self):
        """单位矩阵应返回单位四元数"""
        R = np.eye(3)
        result = af.rot_to_quat(R)
        expected = np.array([1, 0, 0, 0])
        assert_array_almost_equal(result, expected)
    
    def test_quat_roundtrip(self):
        """四元数->旋转矩阵->四元数 往返测试"""
        quat_orig = np.array([0.5, 0.5, 0.5, 0.5])
        quat_orig = quat_orig / np.linalg.norm(quat_orig)
        R = af.quat_to_rot(quat_orig)
        quat_back = af.rot_to_quat(R)
        # 四元数q和-q表示相同旋转
        assert np.allclose(quat_orig, quat_back) or np.allclose(quat_orig, -quat_back)


class TestRotInv:
    """测试旋转矩阵求逆"""
    
    def test_rot_inv_identity(self):
        """单位矩阵的逆是自身"""
        R = np.eye(3)
        result = af.rot_inv(R)
        assert_array_almost_equal(result, R)
    
    def test_rot_inv_property(self):
        """R @ R_inv = I"""
        R = af.rotz(np.pi / 4)
        R_inv = af.rot_inv(R)
        assert_array_almost_equal(R @ R_inv, np.eye(3))


class TestSO3Conversions:
    """测试SO3相关转换"""
    
    def test_vec3_to_so3(self):
        """vec3_to_so3应与hat一致"""
        vec = np.array([1, 2, 3])
        result = af.vec3_to_so3(vec)
        expected = af.hat(vec)
        assert_array_almost_equal(result, expected)
    
    def test_so3_to_vec3(self):
        """so3_to_vec3应正确提取向量"""
        vec_orig = np.array([1.5, -2.3, 0.7])
        so3 = af.hat(vec_orig)
        vec_back = af.so3_to_vec3(so3)
        assert_array_almost_equal(vec_back, vec_orig)
    
    def test_so3_roundtrip(self):
        """vec3->so3->vec3 往返测试"""
        vec_orig = np.array([0.5, 1.2, -0.8])
        so3 = af.vec3_to_so3(vec_orig)
        vec_back = af.so3_to_vec3(so3)
        assert_array_almost_equal(vec_back, vec_orig)


class TestAxisAngle:
    """测试轴角表示"""
    
    def test_axis_angle_basic(self):
        """基本轴角转换"""
        vec = np.array([0, 0, np.pi/2])
        axis, angle = af.axis_angle(vec)
        assert_allclose(angle, np.pi/2)
        assert_array_almost_equal(axis, np.array([0, 0, 1]))


class TestMatrixExp3:
    """测试矩阵指数"""
    
    def test_matrix_to_exp3_z_axis(self):
        """绕Z轴旋转的矩阵指数"""
        angle = np.pi / 4
        so3 = af.hat(np.array([0, 0, angle]))
        result = af.matrix_to_exp3(so3)
        expected = af.rotz(angle)
        assert_array_almost_equal(result, expected)


class TestMatrixLog3:
    """测试矩阵对数"""
    
    def test_matrix_log3_basic(self):
        """基本矩阵对数测试"""
        angle = np.pi / 4
        R = af.rotz(angle)
        log_R = af.matrix_log3(R)
        # 对数矩阵应为反对称矩阵
        assert_array_almost_equal(log_R, -log_R.T)



class TestHomogeneousTransform:
    """测试齐次变换矩阵"""
    
    def test_rp_to_trans_identity(self):
        """单位旋转和零平移"""
        R = np.eye(3)
        p = np.array([0, 0, 0])
        result = af.rp_to_trans(R, p)
        expected = np.eye(4)
        assert_array_almost_equal(result, expected)
    
    def test_rp_to_trans_basic(self):
        """基本齐次变换"""
        R = af.rotz(np.pi / 2)
        p = np.array([1, 2, 3])
        result = af.rp_to_trans(R, p)
        assert_array_almost_equal(result[:3, :3], R)
        assert_array_almost_equal(result[:3, 3], p)
        assert_array_almost_equal(result[3, :], [0, 0, 0, 1])
    
    def test_trans_to_rp(self):
        """齐次变换矩阵分解"""
        R_orig = af.rotx(np.pi / 4)
        p_orig = np.array([1, 2, 3])
        trans = af.rp_to_trans(R_orig, p_orig)
        R_back, p_back = af.trans_to_rp(trans)
        assert_array_almost_equal(R_back, R_orig)
        assert_array_almost_equal(p_back, p_orig)
    
    def test_trans_inv_identity(self):
        """单位变换的逆"""
        trans = np.eye(4)
        result = af.trans_inv(trans)
        assert_array_almost_equal(result, trans)
    
    def test_trans_inv_property(self):
        """T @ T_inv = I"""
        R = af.rotz(np.pi / 3)
        p = np.array([1, 2, 3])
        trans = af.rp_to_trans(R, p)
        trans_inv = af.trans_inv(trans)
        assert_array_almost_equal(trans @ trans_inv, np.eye(4))


class TestSE3Conversions:
    """测试SE3相关转换"""
    
    def test_vec6_to_se3_basic(self):
        """基本vec6到se3转换"""
        vec6 = np.array([1, 2, 3, 4, 5, 6])
        result = af.vec6_to_se3(vec6)
        # 检查旋转部分
        assert_array_almost_equal(result[:3, :3], af.hat(vec6[:3]))
        # 检查平移部分
        assert_array_almost_equal(result[:3, 3], vec6[3:])
        # 检查最后一行
        assert_array_almost_equal(result[3, :], [0, 0, 0, 0])
    
    def test_se3_to_vec6_basic(self):
        """基本se3到vec6转换"""
        vec6_orig = np.array([1, 2, 3, 4, 5, 6])
        se3 = af.vec6_to_se3(vec6_orig)
        vec6_back = af.se3_to_vec6(se3)
        assert_array_almost_equal(vec6_back, vec6_orig)
    
    def test_se3_roundtrip(self):
        """vec6->se3->vec6 往返测试"""
        vec6_orig = np.array([0.1, 0.2, 0.3, 1.0, 2.0, 3.0])
        se3 = af.vec6_to_se3(vec6_orig)
        vec6_back = af.se3_to_vec6(se3)
        assert_array_almost_equal(vec6_back, vec6_orig)


class TestAdjointMatrix:
    """测试伴随矩阵"""
    
    def test_adjoint_identity(self):
        """单位变换的伴随矩阵"""
        trans = np.eye(4)
        result = af.adjoint_mat(trans)
        expected = np.eye(6)
        assert_array_almost_equal(result, expected)
    
    def test_adjoint_pure_rotation(self):
        """纯旋转的伴随矩阵"""
        R = af.rotz(np.pi / 4)
        trans = af.rp_to_trans(R, np.array([0, 0, 0]))
        adj = af.adjoint_mat(trans)
        # 对于纯旋转，伴随矩阵是块对角的
        assert_array_almost_equal(adj[:3, :3], R)
        assert_array_almost_equal(adj[3:, 3:], R)
        assert_array_almost_equal(adj[:3, 3:], np.zeros((3, 3)))


class TestScrewAxis:
    """测试螺旋轴"""
    
    def test_screw_to_axis_pure_rotation(self):
        """纯旋转螺旋轴 (h=0)"""
        q = np.array([0, 0, 0])
        s = np.array([0, 0, 1])
        h = 0
        result = af.screw_to_axis(q, s, h)
        expected = np.array([0, 0, 1, 0, 0, 0])
        assert_array_almost_equal(result, expected)
