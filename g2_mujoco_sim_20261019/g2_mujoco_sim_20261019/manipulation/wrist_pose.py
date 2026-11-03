"""Battery-relative palm geometry to G2 arm end targets; never imports GDK."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

SIDES = {'left': 'l', 'right': 'r'}


def transform(value):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-6, rtol=0)):
        raise ValueError('位姿必须为有限、正交、det=+1的4×4刚体变换')
    return matrix.copy()


def pose7(matrix):
    matrix = transform(matrix)
    return np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()].tolist()


def from_pose7(value):
    values = np.asarray(value, dtype=float)
    if values.shape != (7,) or not np.isfinite(values).all() or not np.isclose(np.linalg.norm(values[3:]), 1, atol=1e-6):
        raise ValueError('位姿格式必须为[x,y,z,qx,qy,qz,qw]，四元数归一化')
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(values[3:]).as_matrix()
    result[:3, 3] = values[:3]
    return result


def joint_origin(joint):
    origin = joint.find('origin')
    matrix = np.eye(4)
    if origin is not None:
        matrix[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        matrix[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return matrix


def palm_mounts(urdf_path):
    root = ET.parse(urdf_path).getroot()
    result = {}
    for side, letter in SIDES.items():
        joint = root.find(f"joint[@name='hand_{letter}_hand_base_joint']")
        if (joint is None or joint.get('type') != 'fixed'
                or joint.find('parent').get('link') != f'arm_{letter}_end_link'
                or joint.find('child').get('link') != f'hand_{letter}_base_link'):
            raise ValueError(f'URDF缺少已核对的{side} end→palm固定安装链')
        result[side] = joint_origin(joint)
    return result


def load_profile(path):
    profile = json.loads(Path(path).read_text())['grasp']
    for side in SIDES:
        transform(profile['hands'][side]['T_O_H'])
    return profile


def solve_wrist_targets(T_base_battery, grasp_profile, urdf_path):
    battery = transform(T_base_battery)
    mounts = palm_mounts(urdf_path)
    palms = {s: battery @ transform(grasp_profile['hands'][s]['T_O_H']) for s in SIDES}
    ends = {s: palms[s] @ np.linalg.inv(mounts[s]) for s in SIDES}
    return {'palms': {s: v.tolist() for s, v in palms.items()},
            'ends': {s: v.tolist() for s, v in ends.items()},
            'targets': {s: pose7(v) for s, v in ends.items()}}


def hand_targets(profile, config, urdf_path, opened=False):
    """Map actuator stems to exact hardware names; validate all dependent joints too."""
    joints = {j.get('name'): j for j in ET.parse(urdf_path).getroot().findall('joint')}
    result = {}
    for side, letter in SIDES.items():
        values = profile['hands'][side]['q_grasp_actuator_rad_by_name']
        expected = {f'{letter.upper()}_{stem}_actuator' for stem in profile['stems']}
        if set(values) != expected or len(expected) != 12:
            raise ValueError(f'{side}抓握源必须完整包含12个命名执行器')
        by_name = {}
        for stem in profile['stems']:
            matches = [n for n in config['grasp']['hands'][side]['joint_names'] if n.endswith(f'_hand_{letter}_{stem}_joint')]
            if len(matches) != 1 or matches[0] not in joints:
                raise ValueError(f'无法唯一映射{side}/{stem}')
            value = float(values[f'{letter.upper()}_{stem}_actuator'])
            if opened and (stem.endswith('_mcp') or stem.endswith('_pip')):
                value = 0.
            by_name[matches[0]] = value
        pending = {n: j for n, j in joints.items() if n.startswith('idx') and f'_hand_{letter}_' in n and j.find('mimic') is not None}
        while pending:
            ready = [n for n, j in pending.items() if j.find('mimic').get('joint') in by_name]
            if not ready:
                raise ValueError('URDF mimic链循环或源关节缺失')
            for name in ready:
                mimic = pending.pop(name).find('mimic')
                by_name[name] = by_name[mimic.get('joint')]*float(mimic.get('multiplier', '1'))+float(mimic.get('offset', '0'))
        for name, value in by_name.items():
            limit = joints[name].find('limit')
            if not np.isfinite(value) or not float(limit.get('lower')) <= value <= float(limit.get('upper')):
                raise ValueError(f'视频手指目标超出URDF限位：{name}={value}')
        result[side] = {n: by_name[n] for n in config['grasp']['hands'][side]['joint_names']}
    return result


def estimate_battery(right_end_pose, T_right_end_battery):
    return from_pose7(right_end_pose) @ transform(T_right_end_battery)


def fk(root, q, link):
    by_child={j.find('child').get('link'):j for j in root.findall('joint')}
    joints={j.get('name'):j for j in root.findall('joint')}
    def value(name):
        if name in q: return q[name]
        mimic=joints[name].find('mimic')
        if mimic is None: raise ValueError(f'缺少关节：{name}')
        return value(mimic.get('joint'))*float(mimic.get('multiplier','1'))+float(mimic.get('offset','0'))
    def visit(name):
        if name=='base_link': return np.eye(4)
        joint=by_child[name]; motion=np.eye(4)
        if joint.get('type')!='fixed':
            axis=np.fromstring(joint.find('axis').get('xyz'),sep=' ')
            if joint.get('type')=='prismatic': motion[:3,3]=axis*value(joint.get('name'))
            else: motion[:3,:3]=Rotation.from_rotvec(axis*value(joint.get('name'))).as_matrix()
        return visit(joint.find('parent').get('link'))@joint_origin(joint)@motion
    return visit(link)

