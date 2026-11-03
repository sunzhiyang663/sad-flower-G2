"""OMPL Reeds-Shepp routes: body-forward/reverse, bounded curvature, no crab."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from ompl import base as ob, geometric as og, util as ou
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation
import yaml
from simulation.full_task_config import load_task
from simulation.build_map9_scene import path, pose
from simulation.map9_route_preview import Clearance, occupied_footprint


def route(start, goal, valid, cfg, bounds, forward_only=False):
    space = (ob.DubinsStateSpace(cfg['turning_radius_m']) if forward_only
             else ob.ReedsSheppStateSpace(cfg['turning_radius_m']))
    limits = ob.RealVectorBounds(2)
    for i, (lo, hi) in enumerate(((bounds[0], bounds[1]), (bounds[2], bounds[3]))):
        limits.setLow(i, lo); limits.setHigh(i, hi)
    space.setBounds(limits)
    setup = og.SimpleSetup(space)
    setup.setStateValidityChecker(lambda s: bool(valid(np.array([s.getX(), s.getY(), s.getYaw()]))))
    si = setup.getSpaceInformation()
    si.setStateValidityCheckingResolution(cfg['sample_step_m']/space.getMaximumExtent())
    states = []
    for p in (start, goal):
        s = space.allocState(); s.setX(float(p[0])); s.setY(float(p[1])); s.setYaw(float(p[2]));states.append(s)
    if not valid(start) or not valid(goal):
        raise ValueError(f'Invalid vehicle endpoints {start}, {goal}')
    setup.setStartAndGoalStates(*states)
    planner = og.RRT(si) if forward_only else og.RRTConnect(si)
    planner.setRange(1.0); setup.setPlanner(planner)
    setup.solve(cfg['planning_time_s'])
    if not setup.haveExactSolutionPath():
        raise RuntimeError('OMPL found no exact collision-free vehicle path')
    setup.simplifySolution(cfg['simplify_time_s'])
    solution = setup.getSolutionPath(); points = []
    temp = space.allocState()
    for i in range(solution.getStateCount()-1):
        a, b = solution.getState(i), solution.getState(i+1)
        length = space.distance(a, b); count = int(np.ceil(length/cfg['sample_step_m']))
        def sample(t):
            space.interpolate(a,b,t,temp)
            return np.array([temp.getX(),temp.getY(),temp.getYaw()])
        def subdivide(t0,t1,p0,p1,depth=0):
            delta = p1[:2]-p0[:2]
            dyaw = np.arctan2(np.sin(p1[2]-p0[2]),np.cos(p1[2]-p0[2]))
            angle = p0[2]+dyaw/2
            lateral = abs(delta @ np.array([-np.sin(angle),np.cos(angle)]))
            curve = abs(dyaw)/max(np.linalg.norm(delta),1e-15)
            if depth < 30 and (lateral > 1e-9 or curve > 1.0001/cfg['turning_radius_m']):
                mid=(t0+t1)/2;pm=sample(mid)
                subdivide(t0,mid,p0,pm,depth+1);subdivide(mid,t1,pm,p1,depth+1)
            else: points.append(p0)
        for k in range(count): subdivide(k/count,(k+1)/count,sample(k/count),sample((k+1)/count))
    p = solution.getState(solution.getStateCount()-1)
    points.append([p.getX(),p.getY(),p.getYaw()])
    points = np.asarray(points)
    # Split direction switches: detect segments whose chord contains a cusp and
    # subdivide using OMPL rather than linearly joining incompatible directions.
    return points


def plan_base_path(start, goal, valid, cfg, bounds,
                   forward_only=False, planner='ompl'):
    """Return an (N, 3) array of map-frame poses [x_m, y_m, yaw_rad].

    Keep scene loading, collision checking and route export outside the planner.
    The OMPL backend preserves the existing forward/reverse motion policy.
    """
    if planner == 'ompl':
        return route(start, goal, valid, cfg, bounds, forward_only=forward_only)
    raise ValueError(f'Unsupported planner: {planner!r}; available: ompl')


def main():
    cfg = yaml.safe_load((ROOT/'simulation/map9.yaml').read_text())
    task = load_task()
    planner = task['vehicle'].get('planner', 'ompl')
    if planner != 'ompl':
        raise ValueError(f'Unsupported planner: {planner!r}; available: ompl')
    ou.RNG.setSeed(task['vehicle']['random_seed'])
    out = path(task['output']); out.mkdir(parents=True, exist_ok=True)
    source = path(cfg['output'])
    meta = json.loads((path(cfg['snapshot'])/'map.json').read_text())
    points = {p['id']: p for p in meta['guide_points']}
    r = meta['grid']['resolution_m']; origin = np.array(meta['grid']['origin']['position'][:2])
    grid = np.load(source/'scene_map.npz')['data']
    model = mujoco.MjModel.from_binary_path(str(source/'scene.mjb'))
    saved = dict(np.load(source/'state.npz'))
    removed = []
    for g in range(model.ngeom):
        if not model.geom(g).name.startswith('map_voxel_'): continue
        p = model.geom_pos[g]; ix,iy = np.floor((p[:2]-origin)/r).astype(int)
        keep = 0 <= iy < grid.shape[0] and 0 <= ix < grid.shape[1] and grid[iy,ix] == 100
        keep &= task['cloud']['retain_height_m'][0] <= p[2] <= task['cloud']['retain_height_m'][1]
        if not keep:
            removed.append(dict(name=model.geom(g).name,centre=p.tolist()))
            model.geom_group[g] = 5;model.geom_contype[g] = 0;model.geom_conaffinity[g] = 0
    (out/'cloud_cleanup.json').write_text(json.dumps(dict(removed=removed,
        policy='Offline only: retain occupied columns within height band; original map unchanged'),indent=2))
    mujoco.mj_saveModel(model,str(out/'scene.mjb'))
    np.savez_compressed(out/'state.npz', **saved)
    footprint = np.array(yaml.safe_load((source/'native_filtered.yaml').read_text())['robot']['footprint'])
    distance = distance_transform_edt(grid == 0)*r
    pickup=np.load(out/'pickup.npz') if (out/'pickup.npz').exists() else None
    if pickup is None:
        from simulation.prepare_map9_manipulation import default_joints
        from simulation.pose_grasp_check import Checker
        plan=json.loads((path(cfg['carry_reference'])/'source_plan.json').read_text())
        inputs=yaml.safe_load((path(cfg['carry_reference'])/'source_inputs.yaml').read_text())['inputs']
        state=Checker(plan['config'],inputs['scene'])
        state.set_state(default_joints(plan['config'],inputs['initial_joints']),np.eye(4))
        default_q=state.data.qpos.copy()
    else:
        default_q=pickup['default_qpos']
        saved['qpos']=pickup['qpos'][-1].copy();saved['T_base_payload']=pickup['payload'][-1].copy()
        saved['initial_base']=pose(points[4],cfg['base_height_m'])
        if 'base_offsets' in pickup:saved['initial_base']=saved['initial_base']@pickup['base_offsets'][-1]
        np.savez_compressed(out/'state.npz',**saved)
    reports = {}
    for a,b,loaded in ((task['start_guide_id'],4,False),*((4,g,True) for g in task['goal_guide_ids'])):
        data = dict(saved);data['qpos'] = pickup['qpos'][-1] if loaded and pickup is not None else saved['qpos'] if loaded else default_q
        if loaded and pickup is not None:data['T_base_payload']=pickup['payload'][-1]
        checker = Clearance(model,data,cfg['base_height_m'],.1)
        checked = 0
        def valid(p):
            nonlocal checked
            x,y = np.floor((p[:2]-origin)/r).astype(int)
            if not (0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]) or grid[y,x] != 0: return False
            if distance[y,x] < .52 and occupied_footprint(p,footprint,.01,r,origin,grid):return False
            # Payload excluded for empty traversal; environment tests remain 3D.
            values,_ = checker.check(p,include_payload=loaded)
            checked += 1
            return values.min() >= task['motion']['min_navigation_clearance_m']
        def p2(k):
            T = pose(points[k]);return np.r_[T[:2,3],Rotation.from_matrix(T[:3,:3]).as_euler('xyz')[2]]
        def pickup_station(index):
            T=pose(points[4])
            if pickup is not None and 'base_offsets' in pickup:T=T@pickup['base_offsets'][index]
            return np.r_[T[:2,3],Rotation.from_matrix(T[:3,:3]).as_euler('xyz')[2]]
        print(f'Planning G{a} -> G{b}, loaded={loaded}',flush=True)
        if loaded:
            exit_pose=p2(a).copy();exit_pose[:2]+=task['vehicle']['rack_exit_map_offset_m']
            exit_pose[2]=task['vehicle']['rack_exit_yaw_rad']
            first = plan_base_path(pickup_station(-1),exit_pose,valid,task['vehicle'],cfg['work_bounds_xy_m'],planner=planner)
            destination=p2(b)
            if task['motion'].get('preposition',{}).get('mode')=='cross_wrap':
                destination[:2]-=task['motion']['table_station_backoff_m']*np.array([np.cos(destination[2]),np.sin(destination[2])])
            onward = plan_base_path(exit_pose,destination,valid,task['vehicle'],cfg['work_bounds_xy_m'],forward_only=True,planner=planner)
            poses = np.vstack([first[:-1],onward])
        else:
            poses = plan_base_path(p2(a),pickup_station(0),valid,task['vehicle'],cfg['work_bounds_xy_m'],forward_only=True,planner=planner)
        if not all(valid(p) for p in poses):raise RuntimeError('Final vehicle samples rejected')
        ds = np.linalg.norm(np.diff(poses[:,:2],axis=0),axis=1)
        dyaw = np.arctan2(np.sin(np.diff(poses[:,2])),np.cos(np.diff(poses[:,2])))
        heading = poses[:-1,2]+dyaw/2
        delta = np.diff(poses[:,:2],axis=0)
        longitudinal = np.sum(delta*np.c_[np.cos(heading),np.sin(heading)],axis=1)
        lateral = np.sum(delta*np.c_[-np.sin(heading),np.cos(heading)],axis=1)
        np.savez_compressed(out/f'route_G{a}_G{b}.npz',poses=poses,ds=ds,dyaw=dyaw,
                            longitudinal=longitudinal,lateral=lateral,loaded=loaded)
        reports[f'G{a}_G{b}'] = dict(samples=len(poses),length_m=float(ds.sum()),validity_checks=checked,
            max_lateral_step_m=float(np.abs(lateral).max()),
            max_curvature_per_m=float(np.max(np.abs(dyaw[ds>1e-6])/ds[ds>1e-6])),
            reverse_distance_m=float(ds[longitudinal<0].sum()),loaded=loaded)
        (out/'vehicle_routes.json').write_text(json.dumps(reports,indent=2))
        print(reports[f'G{a}_G{b}'],flush=True)


if __name__ == '__main__': main()
