"""Frozen camera observations to base_link, without connecting to GDK."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from manipulation.wrist_pose import transform
from manipulation.pose_grasp_geometry import corners


def object_basis(value, handedness):
    """Declared vendor basis only; robot/camera/planning poses remain right handed."""
    if handedness == 'right':
        return transform(value)
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), -1, atol=1e-6, rtol=0)):
        raise ValueError('已声明object_frame_handedness=left，两个物体系变换均须正交且det=-1；请核对视觉实际矩阵')
    return matrix.copy()


def apply_pose_file(inputs, path):
    """手填矩阵交接；保留实测相机外参、时间戳和机器人构型。"""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError('位姿文件须为T_camera_battery或T_camera_object/T_object_battery映射')
    if set(raw) == {'T_camera_battery'}:
        if raw['T_camera_battery'] is None:
            raise ValueError('位姿文件T_camera_battery尚未填写')
        camera_battery = transform(raw['T_camera_battery'])
    elif set(raw) in ({'T_camera_object', 'T_object_battery'},
                      {'T_camera_object', 'T_object_battery', 'object_frame_handedness'}):
        handedness = raw.get('object_frame_handedness', 'right')
        if handedness not in ('left', 'right'):
            raise ValueError('object_frame_handedness仅接受left或right')
        for key in ('T_camera_object', 'T_object_battery'):
            if raw[key] is None:
                raise ValueError(f'位姿文件{key}尚未填写；坐标轴未确认时不要猜填')
        camera_battery = transform(object_basis(raw['T_camera_object'], handedness)
                                  @ object_basis(raw['T_object_battery'], handedness))
    else:
        raise ValueError('位姿文件仅接受T_camera_battery，或T_camera_object/T_object_battery及可选object_frame_handedness')
    if not isinstance(inputs.get('camera_observation'), dict):
        raise ValueError('须先通过--params提供现场camera_observation外参及采集状态')
    result = deepcopy(inputs)
    result['T_base_battery'] = None
    result['camera_observation']['T_camera_battery'] = camera_battery.tolist()
    return result


def resolve_battery_pose(inputs, size):
    observation = inputs.get('camera_observation')
    direct = inputs.get('T_base_battery')
    if (observation is None) == (direct is None):
        raise ValueError('须且仅须提供 T_base_battery 或 camera_observation')
    if observation is None:
        battery = transform(direct)
        info = {'source': 'T_base_battery'}
    else:
        if not observation.get('camera_frame'):
            raise ValueError('inputs.camera_observation.camera_frame须为识别所用的相机坐标系')
        for key in ('T_base_camera', 'T_camera_battery'):
            if observation.get(key) is None:
                raise ValueError(f'inputs.camera_observation.{key} 尚未配置')
        battery = transform(observation['T_base_camera']) @ transform(observation['T_camera_battery'])
        info = {'source': 'frozen_camera_observation', 'camera_frame': observation['camera_frame'],
                'capture_timestamp_ns': observation.get('capture_timestamp_ns'),
                'transform_convention': 'T_A_B maps coordinates in B into A',
                'synchronization_verified': False}
    height = inputs.get('height_reference')
    if height is not None:
        if height.get('T_floor_base') is None:
            raise ValueError('inputs.height_reference.T_floor_base 尚未配置')
        floor_battery = transform(height['T_floor_base']) @ battery
        point = height['point']
        if point not in ('centre', 'bottom'):
            raise ValueError('height_reference.point须为centre或bottom')
        measured, tolerance = height['height_m'], height['tolerance_m']
        if not np.isfinite([measured, tolerance]).all() or min(measured, tolerance) <= 0:
            raise ValueError('height_reference高度/容差须为有限正数')
        actual = float(floor_battery[2, 3] if point == 'centre' else corners(floor_battery, size)[:, 2].min())
        if abs(actual-measured) > tolerance:
            raise ValueError(f'相机位姿与离地高度不符：{point}={actual:.4f}m，配置={measured:.4f}m')
        info['height_check'] = {'point': point, 'height_floor_m': actual,
                                'expected_m': measured, 'tolerance_m': tolerance}
    info['T_base_battery'] = battery.tolist()
    return battery, info
