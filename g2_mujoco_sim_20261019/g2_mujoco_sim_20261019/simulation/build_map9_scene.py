"""Map 9 -> offline MuJoCo scene, synthetic lidar and native ROS navigation config.

No GDK imports, services or hardware commands. Reuses the extraction URDF loader.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from scipy.ndimage import label
from scipy.spatial.transform import Rotation
import yaml


def path(value):
    return ROOT / value


def pose(point, z=0.):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(point['orientation_xyzw']).as_matrix()
    T[:3, 3] = point['position']
    T[2, 3] = z
    return T


def move(points, T):
    return np.asarray(points) @ T[:3, :3].T + T[:3, 3]


def inside_xy(points, bounds):
    x0, x1, y0, y1 = bounds
    return ((points[:, 0] >= x0) & (points[:, 0] <= x1)
            & (points[:, 1] >= y0) & (points[:, 1] <= y1))


def lidar_scan(model, data, T_base, extrinsic, cfg):
    value = json.loads(Path(extrinsic).read_text())
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([value['rotation'][a] for a in 'xyzw']).as_matrix()
    T[:3, 3] = [value['translation'][a] for a in 'xyz']
    T = T_base @ T
    az, el = np.meshgrid(np.linspace(-np.pi, np.pi, cfg['azimuth_samples'], endpoint=False),
                        np.deg2rad(np.linspace(*cfg['elevation_deg'], cfg['elevation_samples'])))
    rays = np.c_[np.cos(el).ravel()*np.cos(az).ravel(),
                 np.cos(el).ravel()*np.sin(az).ravel(), np.sin(el).ravel()]
    directions = np.ascontiguousarray(rays @ T[:3, :3].T)
    ids = np.full(len(rays), -1, dtype=np.int32)
    distances = np.full(len(rays), -1., dtype=np.float64)
    # Ideal post-self-filter sensor: optionally keep visual-mesh body occlusion.
    # Installed lidar origins intersect this URDF's base visual mesh, so the
    # default diagnostic omits body occlusion and records that approximation.
    mujoco.mj_multiRay(model, data, T[:3, 3].copy(), directions.ravel(),
        np.array([0, int(cfg['robot_occlusion']), 1, 1, 1, 0], dtype=np.uint8), True, -1,
        ids, distances, None, len(rays), cfg['max_range_m'])
    valid = (ids >= 0) & (distances > .03) & (distances <= cfg['max_range_m'])
    return move(T[:3, 3]+directions[valid]*distances[valid, None], np.linalg.inv(T_base)), ids[valid]


def filter_payload(points, T_base_payload, half_size, margin):
    """Geometric filter; does not use the raycast's ground-truth semantic labels."""
    local = move(points, np.linalg.inv(T_base_payload))
    return np.all(np.abs(local) <= np.asarray(half_size)+margin, axis=1)


def native_config(cfg, out, meta, spheres):
    """Resolve vendor YAML includes and change only an isolated ROS simulation copy."""
    from utils.config import merge
    nav = path(cfg['native_navigation']['root'])
    def load(file):
        value = yaml.safe_load(file.read_text())
        def resolve(v):
            if isinstance(v, str):
                m = re.fullmatch(r'\$\{yaml:([^{}]+)\}', v)
                if m:
                    return load((file.parent/m[1]).resolve())
                return re.sub(r'\$\{path:([^{}]+)\}',
                    lambda m: str((file.parent/m[1]).resolve()), v)
            if isinstance(v, list):
                return [resolve(x) for x in v]
            if isinstance(v, dict):
                base, local = {}, {}
                for k, x in v.items():
                    if k.startswith('${include'):
                        base = merge(base, resolve(x))
                    else:
                        local[k] = resolve(x)
                return merge(base, local)
            return v
        return resolve(value)
    app = load(nav/'share/bringup/params/g02/app3_sim.yaml')
    # This executable uses ROS bridges only, unlike quark_navigation and its DDS plugins.
    app['logger']['log_path'] = str(out/'native_logs')
    robot = app['robot']
    description = path(cfg['native_navigation']['robot_description'])
    robot.update(urdf_root=str(description.parent)+'/', urdf_dir=description.name+'/urdf/',
                 urdf_type='G2_t2_crsB', end_model='o12_t2')
    ros = app['ros2_bridge_node']
    ros['ros_parameters']['use_sim_time'] = False
    # We integrate cmd_vel in the bridge; do not enable a second chassis simulator.
    for name in ('isw_cart_ros_node', 'isw_cart_sim_ros_node', 'task_service_client_ros_node',
                 'map_service_client_ros_node', 'localization_client_ros_node',
                 'remote_control_ros_node', 'fake_perception_ros_node', 'map_generator_ros_node'):
        ros[name]['enabled'] = False
    app['joystick_node']['enabled'] = False
    ros['perception_ros_node']['lidars'] = [dict(name='livox_front', topic='/livox_front'),
                                          dict(name='livox_back', topic='/livox_back')]
    for name in ('costmap', 'filtered_costmap', 'static_costmap'):
        app['costmap_node'][name]['static_layer']['init_map'] = str(out/'map.yaml')
    # Separate geometric collision cover, not a blanket front-of-robot crop.
    app['robot']['sphere_decomposition']['custom_spheres'].append(
        dict(link='base_link', spheres=spheres, is_end_effector=False))
    required = sorted({p['path'] for p in app['node_lib_manager']['dynamic_libs']}
                      | {p['path'] for p in ros['node_lib_manager']['dynamic_libs']})
    missing = [p for p in required if not Path(p).is_file()]
    for mode in ('raw', 'filtered'):
        value = deepcopy(app)
        if mode == 'raw':
            value['robot']['sphere_decomposition']['custom_spheres'].pop()
        # Synthetic scan has already removed the robot body. The native chain still runs.
        value['logger']['log_path'] = str(out/('native_logs_'+mode))
        (out/('native_'+mode+'.yaml')).write_text(yaml.safe_dump(value, sort_keys=False))
    return dict(binary=str(nav/'bin/genie-navigation'), interface='native ROS /goal_pose2, /cmd_vel, /odom, /livox_front, /livox_back',
                isolated_network_required=True, gdk_pnc_called=False,
                status='blocked_missing_vendor_plugins' if missing else 'prepared_not_executed',
                missing_libraries=missing,
                deployable=False,
                filter_collision_coupling_note='Native custom spheres may also affect RobotFilter deduction; independent perception/collision validation required before deployment.',
                config_entry=str(out/'native_filtered.yaml'))


def build(config, export_only=False):
    from simulation.pose_grasp_check import Checker
    cfg = yaml.safe_load(config.read_text())
    source, out = path(cfg['snapshot']), path(cfg['output'])
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((source/'map.json').read_text())
    grid = np.load(source/'grid.npz')['data']
    if np.linalg.norm(np.array(meta['grid']['origin']['orientation_xyzw'])-[0, 0, 0, 1]) > 1e-8:
        raise ValueError('Rotated map grid requires explicit registration')
    r = meta['grid']['resolution_m']
    origin = np.array(meta['grid']['origin']['position'][:2])
    points = {p['id']: p for p in meta['guide_points']}
    raw = np.load(source/'nonground_cloud.npz')['xyz']
    finite = np.isfinite(raw).all(axis=1)
    local = raw[finite & inside_xy(raw, cfg['work_bounds_xy_m'])].copy()
    up = -np.array(meta['gravity']); up /= np.linalg.norm(up)
    heights = local @ up
    bins = np.arange(*cfg['cloud']['floor_normal_offset_search_m'], cfg['cloud']['floor_histogram_bin_m'])
    hist, edges = np.histogram(heights, bins=bins)
    floor = float(edges[hist.argmax()]+cfg['cloud']['floor_histogram_bin_m']/2)
    local[:, 2] = heights-floor
    # XY remains in the captured navigation grid; this height flattening is not a rigid TF.
    clean = cfg['cleanup']
    selected = inside_xy(local, clean['bounds_xy_m'])
    selected &= (local[:, 2] >= clean['height_range_m'][0]) & (local[:, 2] <= clean['height_range_m'][1])
    if not clean['confirmed_by_user']:
        selected[:] = False
    cloud = local[~selected]
    edited = grid.copy()
    components, _ = label(grid == 100, structure=np.ones((3, 3)))
    yy, xx = np.indices(grid.shape)
    xy = np.c_[xx.ravel(), yy.ravel()]*r+origin+r/2
    roi = inside_xy(xy, clean['bounds_xy_m']).reshape(grid.shape)
    cleared_ids = []
    if clean['confirmed_by_user']:
        for component in np.unique(components[roi & (components > 0)]):
            mask = components == component
            if clean['component_min_cells'] <= mask.sum() <= clean['component_max_cells'] and np.all(roi[mask]):
                edited[mask] = 0
                cleared_ids.append(int(component))
    np.savez_compressed(out/'scene_map.npz', data=edited, cloud=cloud, removed_points=local[selected])
    # Standard ROS map: PGM rows are top down, unknown 205, free 254, occupied 0.
    image = np.where(edited < 0, 205, np.where(edited == 0, 254, 0)).astype(np.uint8)[::-1]
    (out/'map.pgm').write_bytes(f'P5\n{image.shape[1]} {image.shape[0]}\n255\n'.encode()+image.tobytes())
    (out/'map.yaml').write_text(yaml.safe_dump(dict(image='map.pgm', resolution=r,
        origin=[*origin.tolist(), 0.], negate=0, occupied_thresh=.65, free_thresh=.25)))
    rack = yaml.safe_load((path(cfg['rack_reference'])/'source_inputs.yaml').read_text())['inputs']
    carrying = path(cfg['carry_reference'])
    plan = json.loads((carrying/'source_plan.json').read_text())
    if not plan['full_extraction_trajectory_validated']:
        raise ValueError('Carry source must have accepted full extraction')
    replay = np.load(carrying/'replay.npz')
    if str(replay['stages'][-1]) != 'lift' or str(replay['kinds'][-1]) != 'checked':
        raise ValueError('Carry reference must end at checked lift')
    T_base = pose(points[4], cfg['base_height_m'])
    scene = deepcopy(rack['scene'])
    boxes = []
    for name, item in [('support', scene['support'])]+[(x['name'], x) for x in scene['obstacles']]:
        T = T_base @ np.array(item['T_base'])
        item['T_base'] = T.tolist()
        boxes.append(dict(name=name, T=T.tolist(), size=item['size_m'], kind='rack'))
    checker = Checker(plan['config'], scene, include_visuals=True)
    spec = checker.spec
    spec.worldbody.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[35, 35, .1], rgba=[.85, .87, .89, 1], group=4)
    spec.worldbody.add_light(pos=[3, 10, 10], dir=[0, 0, -1], diffuse=[.8, .8, .8])
    for guide in cfg['table']['guide_ids']:
        T = pose(points[guide]); T[:3, 3] = move([[cfg['table']['centre_forward_m'], 0, 0]], T)[0]
        body = spec.worldbody.add_body(name=f'table_G{guide}', pos=T[:3, 3],
            quat=Rotation.from_matrix(T[:3, :3]).as_quat(scalar_first=True))
        length, width, height = cfg['table']['size_m']
        thick = cfg['table']['top_thickness_m']
        # Long edge across approach direction, depth 0.80 m.
        body.add_geom(name=f'table_top_G{guide}', type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0, 0, height-thick/2], size=[width/2, length/2, thick/2], group=2, rgba=[.66, .44, .24, 1])
        for i, (x, y) in enumerate(((-width/2+.06, -length/2+.06),(-width/2+.06, length/2-.06),
                                    (width/2-.06, -length/2+.06),(width/2-.06, length/2-.06))):
            body.add_geom(name=f'table_leg_G{guide}_{i}', type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[x, y, (height-thick)/2], size=[.025, .025, (height-thick)/2], group=2, rgba=[.3, .35, .4, 1])
        Tb = T.copy(); Tb[2, 3] = height/2
        boxes.append(dict(name=f'table_G{guide}', T=Tb.tolist(), size=[width, length, height], kind='table', placement='assumed offset from guide pose'))
    low, high = cfg['cloud']['height_range_m']; v = cfg['cloud']['voxel_m']
    samples = cloud[(cloud[:, 2] >= low) & (cloud[:, 2] <= high)]
    voxels = np.unique(np.floor(samples/v).astype(int), axis=0)*v+v/2
    # Suppress scanned duplicates only inside the explicit rack/table models.
    keep = np.ones(len(voxels), dtype=bool)
    for b in boxes:
        p = move(voxels, np.linalg.inv(b['T']))
        keep &= ~np.all(np.abs(p) <= np.array(b['size'])/2+v/2, axis=1)
    voxels = voxels[keep]
    for i, p in enumerate(voxels):
        spec.worldbody.add_geom(name=f'map_voxel_{i}', type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=p, size=[v*.48]*3, group=4, rgba=[.31, .43, .52, .8])
    model = spec.compile(); data = mujoco.MjData(model)
    model.vis.global_.offwidth = 1280; model.vis.global_.offheight = 720
    model.vis.quality.offsamples = 0
    model.body_pos[model.body('base_link').id] = T_base[:3, 3]
    model.body_quat[model.body('base_link').id] = Rotation.from_matrix(T_base[:3, :3]).as_quat(scalar_first=True)
    data.qpos[:] = replay['qpos'][-1]
    T_payload = np.eye(4)
    T_payload[:3, 3] = replay['mocap_pos'][-1, 0]
    T_payload[:3, :3] = Rotation.from_quat(replay['mocap_quat'][-1, 0], scalar_first=True).as_matrix()
    world_payload = T_base @ T_payload
    data.mocap_pos[0] = world_payload[:3, 3]
    data.mocap_quat[0] = Rotation.from_matrix(world_payload[:3, :3]).as_quat(scalar_first=True)
    for g in range(model.ngeom):
        name = model.body(int(model.geom_bodyid[g])).name
        if name in checker.robot_bodies and model.geom_group[g] in (0, 1):
            model.geom_rgba[g] = [.74, .80, .83, 1] if not name.startswith('arm_') else [.1, .48, .52, 1]
        elif name == 'payload':
            model.geom_group[g] = 3; model.geom_rgba[g] = [1., .65, .12, 1]
        elif name.startswith('scene_'):
            model.geom_group[g] = 2; model.geom_rgba[g] = [.4, .48, .55, 1]
    mujoco.mj_forward(model, data)
    if export_only:
        return model, spec
    mujoco.mj_saveModel(model, str(out/'scene.mjb'))
    np.savez_compressed(out/'state.npz', qpos=data.qpos, T_base_payload=T_payload, initial_base=T_base)
    lidar_reports = {}
    robot_geoms = [g for g in range(model.ngeom)
                   if model.body(int(model.geom_bodyid[g])).name in checker.robot_bodies
                   and model.geom_group[g] in (0, 1)]
    for station, T_scan in [('G4', T_base), ('G6', pose(points[6], cfg['base_height_m']))]:
        # G6 is a separate filter diagnostic, not an accepted route from G4.
        model.body_pos[model.body('base_link').id] = T_scan[:3, 3]
        model.body_quat[model.body('base_link').id] = Rotation.from_matrix(T_scan[:3, :3]).as_quat(scalar_first=True)
        Tb = T_scan @ T_payload
        data.mocap_pos[0] = Tb[:3, 3]
        data.mocap_quat[0] = Rotation.from_matrix(Tb[:3, :3]).as_quat(scalar_first=True)
        mujoco.mj_forward(model, data)
        statistics = {}
        for name in ('front', 'back'):
            xyz, ids = lidar_scan(model, data, T_scan, cfg['lidar'][name+'_extrinsic'], cfg['lidar'])
            robot_hits = np.isin(ids, robot_geoms)
            p = xyz[~robot_hits]; ids = ids[~robot_hits]
            removed = filter_payload(p, T_payload, np.array(cfg['payload']['size_m'])/2, cfg['payload']['filter_margin_m'])
            payload_hits = ids == model.geom('payload').id
            prefix = ('' if station == 'G4' else station+'_')+name
            np.savez_compressed(out/(prefix+'_scan.npz'), raw_base=p.astype(np.float32),
                filtered_base=p[~removed].astype(np.float32), removed=removed, hit_payload=payload_hits)
            # GDK/PointCloud2 byte convention; zero timestamp is synthetic.
            for mode, values in [('raw', p), ('filtered', p[~removed])]:
                stem=prefix+'_'+mode
                (out/(stem+'.bin')).write_bytes(np.asarray(values, dtype='<f4').tobytes())
                (out/(stem+'.json')).write_text(json.dumps(dict(synthetic=True, frame_id='base_link',
                    diagnostic_base_guide=station, timestamp_ns=0, width=len(values), height=1,
                    point_step=12, row_step=12*len(values), is_bigendian=False,
                    fields=[dict(name=a,offset=i*4,datatype=7,count=1) for i,a in enumerate('xyz')]),indent=2))
            external = int((removed & ~payload_hits).sum())
            missed = int((payload_hits & ~removed).sum())
            statistics[name] = dict(all_hits=len(xyz), robot_removed=int(robot_hits.sum()),
                payload_hits=int(payload_hits.sum()), filtered=int(removed.sum()),
                external_points_removed=external, missed_payload_points=missed,
                filter_diagnostic_passed=(external == 0 and missed == 0 and int(payload_hits.sum()) > 0))
        lidar_reports[station] = statistics
    # Eight covering spheres exactly cover a 2x2x2 partition, then add clearance.
    size = np.array(cfg['payload']['size_m']); spheres = []
    for sign in np.ndindex(2, 2, 2):
        c = (np.array(sign)*2-1)*size/4
        spheres.append(dict(center=move([c], T_payload)[0].tolist(),
                            radius=float(np.linalg.norm(size/4)+cfg['payload']['collision_margin_m'])))
    native = native_config(cfg, out, meta, spheres)
    (out/'missing_plugins.json').write_text(json.dumps(dict(
        purpose='Vendor native ROS navigation simulation dependencies; offline scene is usable without them.',
        architecture='aarch64', robot='G2_t2_crsB_o12_t2', variant='g02',
        native_binary=native['binary'],
        native_binary_sha256=hashlib.sha256(Path(native['binary']).read_bytes()).hexdigest(),
        required_libraries=[dict(name=Path(p).name, expected_path=p) for p in native['missing_libraries']],
        request='Provide ROS simulation plugins and their transitive dependencies matching this installed navigation build. No online service replacement has been performed.'),ensure_ascii=False,indent=2))
    report = dict(scope='offline 3D scene + raycast lidar + native ROS navigation preparation',
        hardware_commands_sent=False, source_map_id=meta['map_id'], source_cloud_points=len(raw),
        cleanup=dict(bounds_xy_m=clean['bounds_xy_m'], removed_cloud_points=int(selected.sum()),
                     removed_grid_cells=int((edited != grid).sum()), component_ids=cleared_ids,
                     original_map_modified=False, user_confirmed=clean['confirmed_by_user']),
        floor=dict(up_vector=up.tolist(), offset_m=floor, mode='height flattening preserving grid XY; not a measured rigid registration'),
        assumptions=['Guide points are base poses; rack relative pose from specified reference.',
                     'Table centre offset 1.10m ahead, placement not surveyed.',
                     'Cloud height flattening, voxel envelopes and angular lidar raster are approximations.',
                     'Default raycast omits robot-body occlusion: ideal post-self-filter sensor, not full Livox emulation. Installed lidar extrinsics intersect the selected URDF base mesh and need verification.',
                     'Rigid payload at accepted offline lift pose, not validated for sustained transport.'],
        rack_reference=str(path(cfg['rack_reference'])), carry_reference=str(carrying),
        table_size_m=cfg['table']['size_m'], voxel_count=len(voxels), lidar=lidar_reports['G4'],
        lidar_transit_diagnostic=dict(guide_id=6, route_to_this_pose_validated=False, lidar=lidar_reports['G6']),
        T_base_payload=T_payload.tolist(), payload_spheres=spheres, native_navigation=native,
        native_navigation_executed=False, boxes=boxes,
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (out/'scene_config.yaml').write_bytes(config.read_bytes())
    np.savez_compressed(out/'visual_cloud.npz', xyz=cloud[::max(1, len(cloud)//35000)], voxels=voxels)
    print(json.dumps({k:report[k] for k in ('cleanup','floor','voxel_count','lidar')}, ensure_ascii=False, indent=2))
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'simulation/map9.yaml')
    build(parser.parse_args().config)
