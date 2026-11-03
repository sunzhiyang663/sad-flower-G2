"""Measured support geometry and straight extraction distances, in base_link."""
from itertools import product
import math
import numpy as np

from manipulation.wrist_pose import transform

CORNERS = np.array(list(product((-1., 1.), repeat=3)))


def box(value):
    size = np.asarray(value['size_m'], dtype=float)
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError('现场box.size_m必须为三个有限正数')
    return transform(value['T_base']), size


def corners(T, size):
    return (CORNERS*size/2) @ T[:3, :3].T + T[:3, 3]


def validate_scene(scene, battery, size, tolerance, *, allow_front_overhang=False, minimum_support=0., require_axis_alignment=True):
    if not isinstance(scene, dict) or set(scene) != {'support', 'opening', 'obstacles'}:
        raise ValueError('inputs.scene须包含support、opening（桌面用null）、obstacles（无障碍用[]）')
    for key in ('T_base', 'size_m'):
        if not isinstance(scene['support'], dict) or scene['support'].get(key) is None:
            raise ValueError(f'inputs.scene.support.{key} 尚未配置')
    support, support_size = box(scene['support'])
    if not np.allclose(support[:3, 2], [0, 0, 1], atol=1e-5):
        raise ValueError('首版仅支持水平支撑面')
    relative = np.linalg.inv(support) @ transform(battery)
    if require_axis_alignment and not np.allclose(relative[:3, :3], np.eye(3), atol=1e-3):
        raise ValueError('首版要求电池 upright，+X/+Y/+Z 与支撑框架同向；-X为出口')
    points = corners(relative, size)
    if abs(points[:, 2].min()-support_size[2]/2) > tolerance:
        raise ValueError('电池底面与实测支撑顶面不一致')
    if (points[:, 0].max() > support_size[0]/2+tolerance
            or np.max(np.abs(points[:, 1])) > support_size[1]/2+tolerance):
        raise ValueError('初始电池底面未完全位于支撑范围内')
    if points[:, 0].min() < -support_size[0]/2-tolerance:
        overlap = min(points[:, 0].max(), support_size[0]/2)-max(points[:, 0].min(), -support_size[0]/2)
        if not allow_front_overhang or overlap < minimum_support or relative[0, 3] < -support_size[0]/2+minimum_support:
            raise ValueError('初始电池底面未完全位于支撑范围内；前沿悬伸需显式允许且中心保留支撑余量')
    opening = scene['opening']
    if opening is not None:
        width, height = float(opening['width_m']), float(opening['height_m'])
        if not math.isfinite(width+height) or min(width, height) <= 0:
            raise ValueError('开口宽高必须为有限正数')
        if np.max(np.abs(points[:, 1])) > width/2 or points[:, 2].max() > support_size[2]/2+height:
            raise ValueError('电池超出货架开口')
    if not isinstance(scene['obstacles'], list):
        raise ValueError('scene.obstacles必须为box列表')
    names = set()
    for obstacle in scene['obstacles']:
        box(obstacle)
        name = obstacle['name']
        if not isinstance(name, str) or not name or name in names or name in {'support', 'battery'}:
            raise ValueError('障碍物名称须唯一，且不能为support/battery')
        names.add(name)
    return support, support_size


def extraction_distances(battery, size, scene, hand_max_x, clearance, minimum_supported_length):
    support, extent = box(scene['support'])
    relative = np.linalg.inv(support) @ transform(battery)
    centre = relative[0, 3]
    edge = -extent[0]/2
    # Move until all of the left hand's insertion envelope is outside the front edge.
    partial = max(0., centre+float(hand_max_x)+clearance-edge)
    full = max(partial, centre+size[0]/2+clearance-edge)
    overlap = min(centre-partial+size[0]/2, extent[0]/2)-max(centre-partial-size[0]/2, edge)
    if overlap < minimum_supported_length:
        raise ValueError(f'托底净空与剩余支撑不兼容：剩余{overlap:.4f}m，要求{minimum_supported_length:.4f}m')
    return {'partial_pull_m': float(partial), 'full_pull_m': float(full), 'remaining_support_m': float(overlap),
            'outward_base': (-support[:3, 0]).tolist(),
            'support_stability_verified': False,
            'note': '剩余重叠长度不是稳定性证明；单右手须人工确认可承受拉出及悬伸载荷'}


def shifted(matrix, displacement):
    result = transform(matrix)
    result[:3, 3] += np.asarray(displacement)
    return result


def sample_poses(start, goal, step_m, step_rad):
    from scipy.spatial.transform import Rotation, Slerp
    start, goal = transform(start), transform(goal)
    angle = (Rotation.from_matrix(goal[:3, :3])*Rotation.from_matrix(start[:3, :3]).inv()).magnitude()
    count = max(1, math.ceil(np.linalg.norm(goal[:3, 3]-start[:3, 3])/step_m), math.ceil(angle/step_rad))
    interpolation = Slerp([0, 1], Rotation.from_matrix(np.stack([start[:3, :3], goal[:3, :3]])))
    for fraction in np.linspace(0, 1, count+1)[1:]:
        result = np.eye(4)
        result[:3, :3] = interpolation(fraction).as_matrix()
        result[:3, 3] = (1-fraction)*start[:3, 3]+fraction*goal[:3, 3]
        yield result
