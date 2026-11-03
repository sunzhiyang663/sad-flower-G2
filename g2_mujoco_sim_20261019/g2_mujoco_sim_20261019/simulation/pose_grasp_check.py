"""Isolated URDF/Mink IK and geometry checker. JSON lines, no GDK imports."""
from copy import deepcopy
from itertools import combinations
from pathlib import Path
import hashlib
import json
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import mink
import numpy as np
from scipy.spatial.transform import Rotation

from manipulation.wrist_pose import transform, joint_origin, from_pose7, pose7, palm_mounts
from manipulation.pose_grasp_geometry import box, CORNERS, sample_poses


def text_vector(values):
    return ' '.join(map(str, np.asarray(values).ravel()))


class Checker:
    def __init__(self, config, scene, include_visuals=False):
        self.cfg = config['pose_grasp']
        self.names = config['manipulation']['joint_names']
        self.hand_names = {s: config['grasp']['hands'][s]['joint_names'] for s in ('left', 'right')}
        path = Path(self.cfg['urdf_path'])
        self.urdf = ET.parse(path).getroot()
        root = deepcopy(self.urdf)
        # Collision meshes are retained. Add convex visual envelopes only where the
        # URDF omitted hand collision geometry (including fingertip marker spheres).
        added = []
        for link in root.findall('link'):
            if include_visuals:
                for index, visual in enumerate(link.findall('visual')):
                    visual.set('name', f'{link.get("name")}_preview_visual_{index}')
            if link.get('name').startswith('hand_') and not link.findall('collision'):
                for visual in link.findall('visual'):
                    collision = ET.SubElement(link, 'collision')
                    for tag in ('origin', 'geometry'):
                        value = visual.find(tag)
                        if value is not None:
                            collision.append(deepcopy(value))
                    added.append(link.get('name'))
            if include_visuals:
                for index, collision in enumerate(link.findall('collision')):
                    collision.set('name', f'{link.get("name")}_preview_collision_{index}')
            if not include_visuals:
                for visual in link.findall('visual'):
                    link.remove(visual)
        for mesh in root.iter('mesh'):
            filename = mesh.get('filename')
            if filename.startswith('package://genie_robot_description/'):
                filename = str(Path(self.cfg['description_path'])/filename.split('package://genie_robot_description/')[1])
            elif not Path(filename).is_absolute():
                filename = str(path.parent/filename)
            if not Path(filename).is_file():
                raise ValueError(f'模型网格缺失：{filename}')
            mesh.set('filename', filename)
        custom = root.find('mujoco')
        if custom is not None:
            root.remove(custom)
        compiler = ET.SubElement(root, 'mujoco')
        ET.SubElement(compiler, 'compiler', discardvisual='false' if include_visuals else 'true', fusestatic='false', balanceinertia='true')
        spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding='unicode'))
        self.robot_bodies = {b.name for b in spec.bodies}
        self.model_joints = {j.get('name'): j for j in root.findall('joint') if j.get('type') != 'fixed'}
        self.mimics = {n: j.find('mimic') for n, j in self.model_joints.items() if j.find('mimic') is not None}
        self.env_names = []
        for name, value in [('support', scene['support'])]+[(v['name'], v) for v in scene['obstacles']]:
            T, size = box(value)
            body = spec.worldbody.add_body(name='scene_'+name, pos=T[:3, 3], quat=Rotation.from_matrix(T[:3, :3]).as_quat(scalar_first=True))
            body.add_geom(name='scene_'+name, type=mujoco.mjtGeom.mjGEOM_BOX, size=size/2)
            self.env_names.append('scene_'+name)
        if scene['opening'] is not None:
            support, extent = box(scene['support'])
            width, height = scene['opening']['width_m'], scene['opening']['height_m']
            thickness = self.cfg['collision_clearance_m']
            for name, centre, size in (
                    ('opening_left', [0, width/2+thickness/2, extent[2]/2+height/2], [extent[0], thickness, height]),
                    ('opening_right', [0, -width/2-thickness/2, extent[2]/2+height/2], [extent[0], thickness, height]),
                    ('opening_roof', [0, 0, extent[2]/2+height+thickness/2], [extent[0], width+2*thickness, thickness])):
                position = support[:3, :3] @ centre+support[:3, 3]
                body = spec.worldbody.add_body(name='scene_'+name, pos=position,
                    quat=Rotation.from_matrix(support[:3, :3]).as_quat(scalar_first=True))
                body.add_geom(name='scene_'+name, type=mujoco.mjtGeom.mjGEOM_BOX, size=np.asarray(size)/2)
                self.env_names.append('scene_'+name)
        battery = spec.worldbody.add_body(name='payload', mocap=True)
        battery.add_geom(name='payload', type=mujoco.mjtGeom.mjGEOM_BOX, size=np.array(list(config['battery']['size'].values()))/2)
        self.spec = spec
        self.model = spec.compile()
        self.base_transform = np.eye(4)
        self.c = mink.Configuration(self.model)
        self.data = self.c.data
        srdf_path = Path(self.cfg['srdf_path'])
        self.disabled = set()
        for pair in ET.parse(srdf_path).getroot().findall('disable_collisions'):
            a, b = pair.get('link1'), pair.get('link2')
            def side(name):
                return 'l' if name.startswith(('hand_l_', 'arm_l_')) else 'r' if name.startswith(('hand_r_', 'arm_r_')) else None
            same_hand_group = (pair.get('reason') == 'Group' and a.startswith('hand_')
                               and b.startswith('hand_') and side(a) == side(b))
            if (pair.get('reason') in ('Adjacent', 'Always') or same_hand_group) and not (side(a) and side(b) and side(a) != side(b)):
                self.disabled.add(frozenset((a, b)))
        # Use the intersection of URDF and GDK limits, never expand either model.
        for group, names in self.names.items():
            for name, interval in zip(names, config['manipulation']['joint_limits_rad'][group]):
                self.narrow_limit(name, interval)
        for side, names in self.hand_names.items():
            for name, interval in zip(names, config['grasp']['hands'][side]['joint_limits_rad']):
                self.narrow_limit(name, interval)
        self.robot = [g for g in range(self.model.ngeom)
                      if self.model.geom_group[g] == 0
                      and self.model.body(int(self.model.geom_bodyid[g])).name in self.robot_bodies]
        self.env = [self.model.geom(n).id for n in self.env_names]
        self.payload = self.model.geom('payload').id
        self.pairs = [(a, b) for a, b in combinations(self.robot, 2) if not self.structural_neighbours(a, b)]
        self.collision_pairs = np.asarray(self.pairs+[(g,e) for g in self.robot for e in self.env]
            +[(self.payload,e) for e in self.env]+[(self.payload,g) for g in self.robot],dtype=int)
        self.collision_limit = mink.CollisionAvoidanceLimit(self.model,
            [([a], [b]) for a, b in self.pairs]+[(self.robot, self.env)],
            minimum_distance_from_collisions=self.cfg['collision_clearance_m']+self.cfg['ik_collision_margin_m'],
            collision_detection_distance=10*self.cfg['collision_clearance_m'])
        self.metadata = {'urdf_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'srdf_sha256': hashlib.sha256(srdf_path.read_bytes()).hexdigest(),
                         'robot_collision_geoms': len(self.robot), 'self_collision_pairs': len(self.pairs),
                         'environment_geometries': self.env_names,
                         'robot_environment_checked': True, 'payload_environment_checked': True,
                         'path_step_m': self.cfg['path_step_m'],
                         'continuous_collision_detection': False, 'dynamic_obstacles_checked': False,
                         'added_hand_envelopes': added, 'backend': 'MuJoCo URDF + Mink/DAQP',
                         'collision_model': 'URDF convex envelopes + vendor SRDF Adjacent,Always and intra-hand Group exclusions; cross-arm pairs always checked',
                         'intra_hand_self_collision_validated': False,
                         'dynamics_validated': False}

    def narrow_limit(self, name, interval):
        joint = self.model.joint(name)
        low, high = max(joint.range[0], interval[0]), min(joint.range[1], interval[1])
        if low > high:
            raise ValueError(f'URDF/GDK限位无交集：{name}')
        self.model.jnt_range[joint.id] = [low, high]
        self.model.jnt_limited[joint.id] = True

    def owner_chain(self, geom):
        chain = []
        body = int(self.model.geom_bodyid[geom])
        while body:
            if self.model.body_jntnum[body]:
                chain.append(body)
            body = int(self.model.body_parentid[body])
        return chain

    def structural_neighbours(self, a, b):
        ca, cb = self.owner_chain(a), self.owner_chain(b)
        if ca == cb:
            return True
        links = frozenset(self.model.body(int(self.model.geom_bodyid[g])).name for g in (a, b))
        return links in self.disabled

    def set_state(self, joints, battery):
        required = {n for names in self.names.values() for n in names} | {n for names in self.hand_names.values() for n in names}
        if set(joints) != required:
            raise ValueError(f'离线状态须完整含22上半身+24手关节；缺少{sorted(required-set(joints))}，多出{sorted(set(joints)-required)}')
        q = self.model.qpos0.copy()
        expanded = dict(joints)
        pending = dict(self.mimics)
        while pending:
            ready = [n for n, mimic in pending.items() if mimic.get('joint') in expanded]
            if not ready:
                raise ValueError('无法展开URDF mimic关节')
            for n in ready:
                mimic = pending.pop(n)
                expanded[n] = expanded[mimic.get('joint')]*float(mimic.get('multiplier', 1))+float(mimic.get('offset', 0))
        for name, value in expanded.items():
            joint = self.model.joint(name)
            if not np.isfinite(value) or not joint.range[0]-1e-8 <= value <= joint.range[1]+1e-8:
                raise ValueError(f'关节超出URDF/GDK交集限位：{name}={value}')
            q[int(joint.qposadr[0])] = value
        self.c.update(q)
        self.set_battery(battery)
        self.validate_fk(expanded)

    def set_base(self, value):
        """Offline scene coordinates stay fixed while the robot root moves."""
        self.base_transform = transform(value)
        root = self.model.body('base_link')
        self.model.body_pos[root.id] = self.base_transform[:3, 3]
        self.model.body_quat[root.id] = Rotation.from_matrix(self.base_transform[:3, :3]).as_quat(scalar_first=True)

    def set_battery(self, battery):
        battery = transform(battery)
        self.battery = battery
        self.data.mocap_pos[0] = battery[:3, 3]
        self.data.mocap_quat[0] = Rotation.from_matrix(battery[:3, :3]).as_quat(scalar_first=True)
        mujoco.mj_forward(self.model, self.data)

    def validate_fk(self, values):
        transforms = {'base_link': self.base_transform.copy()}
        pending = list(self.urdf.findall('joint'))
        while pending:
            ready = [j for j in pending if j.find('parent').get('link') in transforms]
            if not ready:
                raise ValueError('URDF根必须为base_link且树连通')
            for joint in ready:
                pending.remove(joint)
                motion = np.eye(4)
                value = values.get(joint.get('name'), 0.)
                if joint.get('type') in ('revolute', 'continuous', 'prismatic'):
                    axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                    if joint.get('type') == 'prismatic':
                        motion[:3, 3] = value*axis
                    else:
                        motion[:3, :3] = Rotation.from_rotvec(value*axis).as_matrix()
                transforms[joint.find('child').get('link')] = transforms[joint.find('parent').get('link')] @ joint_origin(joint) @ motion
        for name in ('arm_l_end_link', 'arm_r_end_link', 'hand_l_base_link', 'hand_r_base_link'):
            actual = self.c.get_transform_frame_to_world(name, 'body').as_matrix()
            if not np.allclose(actual, transforms[name], atol=1e-6):
                raise ValueError(f'URDF导入FK不一致：{name}')

    def current_joints(self):
        return {n: float(self.data.qpos[int(self.model.joint(n).qposadr[0])])
                for names in list(self.names.values())+list(self.hand_names.values()) for n in names}

    def hand_bounds(self, side, T_battery_end):
        letter = {'left': 'l', 'right': 'r'}[side]
        end = self.c.get_transform_frame_to_world(f'arm_{letter}_end_link', 'body').as_matrix()
        points, slab_points = [], []
        thickness = box(self.scene['support'])[1][2]
        top = -self.model.geom_size[self.payload, 2]
        low, high = top-thickness-self.cfg['collision_clearance_m'], top+self.cfg['collision_clearance_m']
        for g in self.robot:
            if self.model.body(int(self.model.geom_bodyid[g])).name.startswith(f'hand_{letter}_'):
                local = self.model.geom_aabb[g, :3]+CORNERS*self.model.geom_aabb[g, 3:]
                world = local @ self.data.geom_xmat[g].reshape(3, 3).T+self.data.geom_xpos[g]
                T = transform(T_battery_end) @ np.linalg.inv(end)
                transformed = world @ T[:3, :3].T+T[:3, 3]
                points.extend(transformed)
                slab_points.extend(p for p in transformed if low <= p[2] <= high)
                # Clip the envelope edges against the support slab's vertical band.
                for a, b in combinations(range(8), 2):
                    if np.count_nonzero(CORNERS[a] != CORNERS[b]) != 1:
                        continue
                    p, q = transformed[a], transformed[b]
                    if abs(q[2]-p[2]) > 1e-10:
                        for plane in (low, high):
                            alpha = (plane-p[2])/(q[2]-p[2])
                            if 0 <= alpha <= 1:
                                slab_points.append(p+alpha*(q-p))
        return {'min': np.min(points, axis=0).tolist(), 'max': np.max(points, axis=0).tolist(),
                'support_band_max_x': None if not slab_points else float(np.max(slab_points, axis=0)[0])}

    def collisions(self, contact_sides):
        cfg = self.cfg
        clearance = cfg['collision_clearance_m']
        penetration = cfg['contact_penetration_tolerance_m']
        pairs = self.collision_pairs
        a,b=pairs.T
        # Same bounding-sphere rejection, vectorized before convex queries.
        near=np.linalg.norm(self.data.geom_xpos[a]-self.data.geom_xpos[b],axis=1) <= (
            self.model.geom_rbound[a]+self.model.geom_rbound[b]+clearance)
        minimum = float('inf')
        for a, b in pairs[near]:
            segment = np.zeros(6)
            distance = mujoco.mj_geomDistance(self.model, self.data, a, b, clearance+1e-6, segment)
            minimum = min(minimum, float(distance))
            if distance >= clearance:
                continue
            name = self.model.body(int(self.model.geom_bodyid[b])).name
            allowed = False
            if a == self.payload and b == self.model.geom('scene_support').id:
                # Support contact is legal only at its top, never through the slab.
                support = box(self.scene['support'])[0]
                z = ((self.data.geom_xpos[a]-support[:3, 3]) @ support[:3, 2])
                allowed = z > 0 and distance >= -penetration
            elif a == self.payload:
                side = 'left' if name.startswith('hand_l_') else 'right' if name.startswith('hand_r_') else None
                if side in contact_sides and not name.endswith('camera_link') and distance >= -penetration:
                    local = (segment[3:]-self.battery[:3, 3]) @ self.battery[:3, :3]
                    size = self.model.geom_size[self.payload]
                    faces = self.cfg['contact_faces'][side]
                    allowed = any(abs(local['xyz'.index(face[1])]-({'-': -1, '+': 1}[face[0]])*size['xyz'.index(face[1])]) <= clearance+penetration for face in faces)
            if not allowed:
                labels = [self.model.body(int(self.model.geom_bodyid[g])).name+'/'+(self.model.geom(g).name or str(g)) for g in (a, b)]
                raise ValueError(f'碰撞/净空不足：{labels[0]} ↔ {labels[1]}，距离{distance:.5f}m')
        return {'minimum_near_distance_m': None if minimum == float('inf') else minimum}

    def fit_left(self, request):
        return self.fit_hand(request, 'left')

    def fit_hand(self, request, side):
        """Fit palm/position targets to the t2 envelopes; arm IK checked separately."""
        from scipy.optimize import differential_evolution
        fit = self.cfg[side+'_fit']
        letter = 'l' if side == 'left' else 'r'
        original = transform(request['T_battery_palm'])
        mount = palm_mounts(self.cfg['urdf_path'])[side]
        geoms = [g for g in self.robot if self.model.body(int(self.model.geom_bodyid[g])).name.startswith(f'hand_{letter}_')]
        captures = []
        for targets in (request['open_targets'], request['closed_targets']):
            self.set_state({**request['joints'], **targets}, request['battery'])
            end = self.c.get_transform_frame_to_world(f'arm_{letter}_end_link', 'body').as_matrix()
            inverse = np.linalg.inv(end)
            capture = []
            for g in geoms:
                rotation = inverse[:3, :3] @ self.data.geom_xmat[g].reshape(3, 3)
                position = inverse[:3, :3] @ self.data.geom_xpos[g]+inverse[:3, 3]
                local = self.model.geom_aabb[g, :3]+CORNERS*self.model.geom_aabb[g, 3:]
                capture.append((g, position, rotation, local @ rotation.T+position))
            captures.append(capture)
        opposite_capture = []
        opposite = request.get('opposite_grasp')
        if opposite is not None:
            other = 'right' if side == 'left' else 'left'
            other_letter = 'r' if side == 'left' else 'l'
            self.set_state({**request['joints'], **opposite['closed_targets']}, request['battery'])
            other_end = self.c.get_transform_frame_to_world(f'arm_{other_letter}_end_link', 'body').as_matrix()
            desired = self.battery @ transform(opposite['T_battery_palm']) @ np.linalg.inv(palm_mounts(self.cfg['urdf_path'])[other])
            move = desired @ np.linalg.inv(other_end)
            for g in self.robot:
                if self.model.body(int(self.model.geom_bodyid[g])).name.startswith(f'hand_{other_letter}_'):
                    opposite_capture.append((g, move[:3, :3] @ self.data.geom_xpos[g]+move[:3, 3],
                        move[:3, :3] @ self.data.geom_xmat[g].reshape(3, 3)))
        inverse_end = inverse.copy()
        source_closed = dict(request['closed_targets'])
        finger_names = [n for n in source_closed if '_thumb_' not in n and n.endswith(('_mcp_joint', '_pip_joint'))]
        thumb_names = [n for n in source_closed if '_thumb_' in n and n.endswith(('_mcp_joint', '_pip_joint'))]

        def closure_capture(finger, thumb):
            targets = {n: v+(finger if n in finger_names else -thumb if n in thumb_names else 0.) for n, v in source_closed.items()}
            expanded = dict(targets)
            pending = {n: m for n, m in self.mimics.items() if f'_hand_{letter}_' in n}
            while pending:
                ready = [n for n, m in pending.items() if m.get('joint') in expanded]
                if not ready:
                    raise ValueError(f'{side}手mimic链缺失')
                for n in ready:
                    mimic = pending.pop(n)
                    expanded[n] = expanded[mimic.get('joint')]*float(mimic.get('multiplier', 1))+float(mimic.get('offset', 0))
            for n, v in expanded.items():
                joint = self.model.joint(n)
                if not joint.range[0] <= v <= joint.range[1]:
                    return None, targets
                self.data.qpos[int(joint.qposadr[0])] = v
            mujoco.mj_kinematics(self.model, self.data)
            captured = []
            for g in geoms:
                rotation = inverse_end[:3, :3] @ self.data.geom_xmat[g].reshape(3, 3)
                position = inverse_end[:3, :3] @ self.data.geom_xpos[g]+inverse_end[:3, 3]
                local = self.model.geom_aabb[g, :3]+CORNERS*self.model.geom_aabb[g, 3:]
                captured.append((g, position, rotation, local @ rotation.T+position))
            return captured, targets
        half = self.model.geom_size[self.payload]
        slab_bottom = -half[2]-box(self.scene['support'])[1][2]-self.cfg['collision_clearance_m']
        slab_top = -half[2]+self.cfg['collision_clearance_m']
        maximum_x = half[0]-self.cfg['minimum_supported_length_m']-self.cfg['collision_clearance_m']-self.cfg['ik_position_tolerance_m']
        # Reserve the IK position error inside the total contact tolerance.
        tolerance = max(0., self.cfg['contact_penetration_tolerance_m']
                        -max(self.cfg['ik_solve_position_tolerance_m'], self.cfg['grasp_fit_position_reserve_m']))
        if opposite_capture:
            tolerance = min(tolerance, max(0., self.cfg['contact_penetration_tolerance_m']
                                          -self.cfg['paired_contact_penetration_reserve_m']))
        gap = fit['contact_gap_m']
        thumb_face = fit.get('thumb_contact_face', '+y')
        if thumb_face not in ('+y', '-y'):
            raise ValueError('thumb_contact_face须为+y或-y')
        thumb_y = half[1] if thumb_face == '+y' else -half[1]

        def evaluate(parameters, details=False):
            dx, dy, dz, roll, finger, thumb = parameters
            closed_capture, closed_targets = closure_capture(finger, thumb)
            if closed_capture is None:
                return 1e3 if not details else {'accepted': False, 'error': '手部候选超限'}
            palm = original.copy()
            palm[:3, 3] += [dx, dy, dz]
            palm[:3, :3] = Rotation.from_rotvec([roll, 0, 0]).as_matrix() @ original[:3, :3]
            end = palm @ np.linalg.inv(mount)
            envelope_x = -float('inf')
            for capture in (captures[0], closed_capture):
                for g, position, rotation, points in capture:
                    transformed = points @ end[:3, :3].T+end[:3, 3]
                    if transformed[:, 2].min() <= slab_top and transformed[:, 2].max() >= slab_bottom:
                        envelope_x = max(envelope_x, float(transformed[:, 0].max()))
            world = self.battery @ end
            paired_clearance = self.cfg['paired_hand_clearance_m']
            nearest_opposite = paired_clearance
            if opposite_capture:
                other_ids = np.array([g for g, _, _ in opposite_capture])
                other_positions = np.array([p for _, p, _ in opposite_capture])
                for g, position, rotation in opposite_capture:
                    self.data.geom_xpos[g] = position
                    self.data.geom_xmat[g] = rotation.ravel()
                for capture in (captures[0], closed_capture):
                    ids = np.array([g for g, _, _, _ in capture])
                    positions = np.array([world[:3, :3] @ p+world[:3, 3] for _, p, _, _ in capture])
                    for (g, _, rotation, _), position in zip(capture, positions):
                        self.data.geom_xpos[g] = position
                        self.data.geom_xmat[g] = (world[:3, :3] @ rotation).ravel()
                    distances = np.linalg.norm(positions[:, None, :]-other_positions[None, :, :], axis=2)
                    nearby = distances <= self.model.geom_rbound[ids, None]+self.model.geom_rbound[other_ids][None, :]+paired_clearance
                    for a, b in np.argwhere(nearby):
                        nearest_opposite = min(nearest_opposite, float(mujoco.mj_geomDistance(
                            self.model, self.data, ids[a], other_ids[b], paired_clearance, None)))
            metrics = {n: [] for n in ('thumb', 'index', 'middle', 'ring', 'pinky', 'bottom')}
            deepest = 0.
            for g, position, rotation, points in closed_capture:
                self.data.geom_xpos[g] = world[:3, :3] @ position+world[:3, 3]
                self.data.geom_xmat[g] = (world[:3, :3] @ rotation).ravel()
                segment = np.zeros(6)
                distance = float(mujoco.mj_geomDistance(self.model, self.data, self.payload, g, 1., segment))
                deepest = min(deepest, distance)
                surface = (segment[:3]-self.battery[:3, 3]) @ self.battery[:3, :3]
                contact = (segment[3:]-self.battery[:3, 3]) @ self.battery[:3, :3]
                name = self.model.body(int(self.model.geom_bodyid[g])).name
                stem = next((n for n in ('thumb', 'index', 'middle', 'ring', 'pinky') if f'_{n}_' in name), None)
                if stem is not None:
                    target_y = thumb_y if stem == 'thumb' else -thumb_y
                    depth = (max(-half[2]+fit['wrap_depth_m']-contact[2], 0.) if side == 'left'
                             else max(-half[0]+fit['wrap_depth_m']-contact[0], 0.))
                    score = max(distance, 0.)**2+(surface[1]-target_y)**2+depth**2
                    metrics[stem].append(score)
                if not name.endswith('camera_link'):
                    metrics['bottom'].append(max(distance, 0.)**2+(surface[2]+half[2])**2)
            scores = {n: min(values, default=1.) for n, values in metrics.items()}
            fingers = sorted(scores[n] for n in ('index', 'middle', 'ring', 'pinky'))
            # Leave numerical slack; feasible candidates must pass the hard checks too.
            penalty = 10000*(max(-tolerance+.0002-deepest, 0.)**2
                             +(max(envelope_x-maximum_x+.0002, 0.)**2 if side == 'left' else 0.)
                             +(max(paired_clearance-nearest_opposite, 0.)**2 if opposite_capture else 0.))
            objective = penalty+scores['thumb']+sum(fingers[:2])+(scores['bottom'] if side == 'left' else 0.)+.0001*(dx*dx+dy*dy+dz*dz+roll*roll*.01+finger*finger+thumb*thumb)
            if not details:
                return objective
            valid = deepest >= -tolerance and max(scores['thumb'], fingers[1]) <= gap*gap
            if opposite_capture:
                valid = valid and nearest_opposite >= paired_clearance-1e-9
            if side == 'left':
                valid = valid and envelope_x <= maximum_x and scores['bottom'] <= gap*gap
            return {'accepted': bool(valid), 'T_battery_palm': palm.tolist(),
                    'thumb_contact_face': thumb_face,
                    'adjustment': {'translation_battery_m': [float(dx), float(dy), float(dz)], 'roll_battery_rad': float(roll)},
                    'finger_closure_delta_rad': float(finger), 'thumb_closure_delta_rad': float(thumb),
                    'closed_targets': {n: float(v) for n, v in closed_targets.items()},
                    'deepest_penetration_m': float(deepest),
                    'support_band_max_x_m': float(envelope_x) if np.isfinite(envelope_x) else None,
                    'required_max_x_m': float(maximum_x) if side == 'left' else None,
                    'contact_region_errors_m': {n: float(np.sqrt(v)) for n, v in scores.items()},
                    'opposite_hand_checked': bool(opposite_capture),
                    'opposite_hand_min_clearance_m': float(nearest_opposite) if opposite_capture else None,
                    'opposite_hand_clearance_measurement_limit_m': paired_clearance if opposite_capture else None,
                    'geometry_only': True, 'dynamics_validated': False, 'hardware_validated': False}

        bounds = [(-fit['max_outward_shift_m'], 0.), (-fit['lateral_range_m'], fit['lateral_range_m']),
                  (-fit['vertical_range_m'], fit['vertical_range_m']), (-fit['roll_range_rad'], fit['roll_range_rad']),
                  ((0. if side == 'left' else -fit['finger_closure_range_rad']), fit['finger_closure_range_rad']),
                  ((0. if side == 'left' else -fit['thumb_closure_range_rad']), fit['thumb_closure_range_rad'])]
        result = differential_evolution(evaluate, bounds, seed=0, popsize=8, maxiter=fit['max_iterations'], polish=True, tol=1e-7)
        return evaluate(result.x, details=True)

    def solve(self, goals, contacts, attachment=None, allow_waist=False, passive=(), *, constructing=False):
        allowed = {int(self.model.joint(n).dofadr[0]) for side in goals for n in self.names[side]}
        moving_names = [n for side in goals for n in self.names[side]]
        if allow_waist:
            moving_names += self.names['waist']
            allowed.update(int(self.model.joint(n).dofadr[0]) for n in self.names['waist'])
        freeze = mink.DofFreezingTask(self.model, [i for i in range(self.model.nv) if i not in allowed])
        tasks = []
        for side, goal in goals.items():
            frame = f"arm_{'l' if side == 'left' else 'r'}_end_link"
            cost = self.cfg.get('waiting_task_cost', .01) if side in passive else 1.
            task = mink.FrameTask(frame, 'body', position_cost=cost, orientation_cost=cost, lm_damping=.01)
            task.set_target(mink.SE3.from_matrix(transform(goal)))
            tasks.append(task)
        if allow_waist:
            # Default: preserve frontal roll/yaw. Offline branch selection may
            # explicitly free roll as well, keeping yaw constrained.
            torso_cost = self.cfg.get('waist_orientation_cost', [1., 0., 1.])
            torso = mink.FrameTask('body_link5', 'body', position_cost=0.,
                                   orientation_cost=torso_cost, lm_damping=.01)
            target = self.c.get_transform_frame_to_world('body_link5', 'body').as_matrix()
            target[:3, :3] = self.base_transform[:3, :3]
            torso.set_target(mink.SE3.from_matrix(target))
            tasks.append(torso)
        posture = mink.PostureTask(self.model, cost=self.cfg['ik_posture_cost'])
        posture.set_target_from_configuration(self.c)
        repairing_seed_collision = False
        for iteration in range(self.cfg['ik_max_iterations']):
            if attachment is not None:
                right = self.c.get_transform_frame_to_world('arm_r_end_link', 'body').as_matrix()
                self.set_battery(right @ attachment)
            errors = [t.compute_error(self.c) for t in tasks]
            wrist_errors = errors[:len(goals)]
            torso_ok = not allow_waist or np.linalg.norm(errors[-1][3:] * (np.array(torso_cost)>0)) <= self.cfg['ik_solve_orientation_tolerance_rad']
            if torso_ok and all(np.linalg.norm(e[:3]) <= self.cfg['waiting_position_tolerance_m' if side in passive else 'ik_solve_position_tolerance_m']
                                and np.linalg.norm(e[3:]) <= self.cfg['waiting_orientation_tolerance_rad' if side in passive else 'ik_solve_orientation_tolerance_rad']
                                for side, e in zip(goals, wrist_errors)):
                try:
                    self.collisions(contacts)
                except ValueError:
                    if not constructing:
                        raise
                    # Numerical seed construction may start in collision. Keep
                    # optimizing with collision limits until the endpoint clears.
                    repairing_seed_collision = True
                else:
                    return iteration
            limits = [mink.ConfigurationLimit(self.model),
                      mink.VelocityLimit(self.model, {n: self.cfg['joint_speed_rad_s'] for n in moving_names})]
            if not constructing or repairing_seed_collision:
                limits.append(self.collision_limit)
            velocity = mink.solve_ik(self.c, tasks+[posture], self.cfg['ik_dt_s'], 'daqp', damping=1e-5,
                                     constraints=[freeze], limits=limits)
            self.c.integrate_inplace(velocity, self.cfg['ik_dt_s'])
            if attachment is not None:
                right = self.c.get_transform_frame_to_world('arm_r_end_link', 'body').as_matrix()
                self.set_battery(right @ attachment)
            if not constructing:
                self.collisions(contacts)
        residuals = {side: {'position_m': float(np.linalg.norm(task.compute_error(self.c)[:3])),
                            'orientation_rad': float(np.linalg.norm(task.compute_error(self.c)[3:]))}
                     for side, task in zip(goals, tasks)}
        if allow_waist:
            residuals['torso_facing_error_rad'] = float(np.linalg.norm(tasks[-1].compute_error(self.c)[3:] * (np.array(torso_cost)>0)))
            residuals['torso_pitch_rad'] = float(tasks[-1].compute_error(self.c)[4])
        if constructing:
            self.collisions(contacts)
        raise ValueError(f'离线IK未收敛；目标不可达或当前初值下无连续解；残差={residuals}')

    def request(self, request):
        self.set_base(request.get('T_world_base', np.eye(4)))
        self.set_state(request['joints'], request['battery'])
        operation = request['operation']
        contacts = request.get('contacts', [])
        if operation == 'spread_grasp_stage':
            from simulation.spread_grasp_check import check_stage
            return check_stage(self, request)
        if operation == 'fit_left':
            return self.fit_left(request)
        if operation == 'fit_right':
            return self.fit_hand(request, 'right')
        if operation == 'bounds':
            return self.hand_bounds(request['side'], request['T_battery_end'])
        if operation == 'construct_seed':
            # A numerical search for the first posture, NOT a motion from the
            # numerical seed. Always check the final pose with full geometry.
            # Actual motion requests below never use constructing=True.
            self.solve(request['goals'], contacts, constructing=True)
            self.validate_fk(self.current_joints())
            return {'joints': self.current_joints(), 'endpoint_collision_checked': True,
                    'transition_to_seed_validated': False, 'hardware_validated': False}
        self.collisions(contacts)
        if operation in ('waist_motion', 'arm_joint_motion'):
            targets = request['targets']
            group = 'waist' if operation == 'waist_motion' else request['side']
            if group not in ('waist', 'left', 'right') or set(targets) != set(self.names[group]) or contacts or request.get('payload_from_right') is not None:
                raise ValueError('关节插值仅允许空载、完整指定关节组目标')
            start = self.current_joints()
            count = max(1, int(np.ceil(max(abs(v-start[n]) for n, v in targets.items())/self.cfg['path_step_rad'])))
            path = []
            for alpha in np.linspace(0, 1, count+1)[1:]:
                joints = {**start, **{n: start[n]+alpha*(v-start[n]) for n, v in targets.items()}}
                self.set_state(joints, request['battery'])
                self.collisions(contacts)
                path.append({'joints': self.current_joints(), 'targets': {}})
            return {'path': path, 'metadata': self.metadata, 'final_battery': self.battery.tolist(),
                    'interpolation': 'offline linear joint candidate; no controller timing'}
        if operation in ('base_motion','base_push'):
            start = self.base_transform.copy()
            goal = transform(request['goal_base'])
            delta = goal[:3, 3]-start[:3, 3]
            local = start[:3, :3].T @ delta
            if (not np.allclose(start[:3, :3], goal[:3, :3], atol=1e-9)
                    or (local[0] > 1e-9 if operation=='base_motion' else local[0] < -1e-9)
                    or np.linalg.norm(local[1:]) > 1e-9):
                raise ValueError('离线底盘抽出须定向后退，台面推送须定向前进；禁止横移或转向')
            attachment = transform(request['payload_from_right'])
            path = []
            for base in sample_poses(start, goal, self.cfg['path_step_m'], self.cfg['path_step_rad']):
                self.set_base(base)
                self.set_state(request['joints'], request['battery'])
                right = self.c.get_transform_frame_to_world('arm_r_end_link', 'body').as_matrix()
                self.set_battery(right @ attachment)
                self.collisions(contacts)
                path.append({'joints': self.current_joints(), 'targets': {},
                             'T_world_base': base.tolist(), 'battery': self.battery.tolist()})
            return {'path': path, 'metadata': self.metadata,
                    'final_battery': self.battery.tolist(), 'final_base': self.base_transform.tolist()}
        if operation == 'state':
            result = {'metadata': self.metadata, 'joints': self.current_joints(), 'ends': {
                s: pose7(self.c.get_transform_frame_to_world(f'arm_{letter}_end_link', 'body').as_matrix())
                for s, letter in (('left', 'l'), ('right', 'r'))}}
            result['torso_pose'] = pose7(self.c.get_transform_frame_to_world('body_link5', 'body').as_matrix())
            result['shoulder_heights_m'] = {s: float(self.c.get_transform_frame_to_world(f'arm_{letter}_link1', 'body').translation()[2])
                for s, letter in (('left', 'l'), ('right', 'r'))}
            if request.get('ends'):
                for side, value in request['ends'].items():
                    actual = self.c.get_transform_frame_to_world(f"arm_{'l' if side == 'left' else 'r'}_end_link", 'body').as_matrix()
                    expected = from_pose7(value)
                    if (np.linalg.norm(actual[:3, 3]-expected[:3, 3]) > self.cfg['ik_position_tolerance_m']
                            or Rotation.from_matrix(actual[:3, :3].T @ expected[:3, :3]).magnitude() > self.cfg['ik_orientation_tolerance_rad']):
                        raise ValueError(f'{side}实测TF与指定URDF的FK不一致，禁止执行')
            return result
        if operation == 'hand':
            start = self.current_joints()
            change = max(abs(v-start[n]) for n, v in request['targets'].items())
            count = max(1, int(np.ceil(change/self.cfg['hand_path_step_rad'])))
            for alpha in np.linspace(0, 1, count+1)[1:]:
                joints = {**start, **{n: start[n]+alpha*(v-start[n]) for n, v in request['targets'].items()}}
                self.set_state(joints, request['battery'])
                self.collisions(contacts)
            return {'samples': count, 'metadata': self.metadata}
        if operation != 'motion':
            raise ValueError('未知离线检查操作')
        passive = request.get('passive_wait_sides', [])
        if (set(passive)-set(request['goals']) or set(passive) & set(contacts)
                or request.get('payload_from_right') is not None and 'right' in passive):
            raise ValueError('只有未接触负载的目标手可以作为等待区域')
        starts = {s: from_pose7(request['ends'][s]) if request.get('ends') else
                  self.c.get_transform_frame_to_world(f"arm_{'l' if s == 'left' else 'r'}_end_link", 'body').as_matrix()
                  for s in request['goals']}
        samples = {s: list(sample_poses(starts[s], goal, self.cfg['path_step_m'], self.cfg['path_step_rad'])) for s, goal in request['goals'].items()}
        count = max(map(len, samples.values()))
        # Both wrists share fractions so paired translations remain paired.
        from scipy.spatial.transform import Slerp
        interpolations = {s: Slerp([0, 1], Rotation.from_matrix(np.stack([starts[s][:3, :3], transform(g)[:3, :3]]))) for s, g in request['goals'].items()}
        path = []
        attachment = None if request.get('payload_from_right') is None else transform(request['payload_from_right'])
        for fraction in np.linspace(0, 1, count+1)[1:]:
            goals = {}
            for side, goal in request['goals'].items():
                goal = transform(goal)
                if side in passive:
                    goals[side] = goal
                    continue
                T = np.eye(4)
                T[:3, :3] = interpolations[side](fraction).as_matrix()
                T[:3, 3] = (1-fraction)*starts[side][:3, 3]+fraction*goal[:3, 3]
                goals[side] = T
            # A held shell follows the actual FK at every internal IK iterate;
            # moving it to a future waypoint first produces false finger collisions.
            iterations = self.solve(goals, contacts, attachment, request.get('allow_waist', False), passive)
            # Verify mimic and all limits at every accepted configuration.
            actual = self.current_joints()
            if request.get('payload_from_right') is not None:
                right_actual = self.c.get_transform_frame_to_world('arm_r_end_link', 'body').as_matrix()
                self.set_battery(right_actual @ transform(request['payload_from_right']))
            self.set_state(actual, self.battery)
            self.collisions(contacts)
            path.append({'targets': {s: pose7(T) for s, T in goals.items()}, 'joints': actual, 'iterations': iterations})
        return {'path': path, 'metadata': self.metadata, 'final_battery': self.battery.tolist(),
                'passive_wait_sides': passive}


def serve():
    checker = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if checker is None:
                checker = Checker(request['config'], request['scene'])
                checker.scene = request['scene']
                result = checker.metadata
            else:
                result = checker.request(request)
            response = {'ok': True, 'result': result}
        except Exception as error:
            response = {'ok': False, 'error': f'{type(error).__name__}: {error}'}
        print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == '__main__':
    serve()
