"""Offline reference routes and sampled whole-body clearance; no GDK or control.

SciPy supplies graph search. A fixed-yaw polygon chassis mask generates reference
translations; every interpolated pose, including the final turn, is independently
checked against the map and the MuJoCo robot/payload geometry.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation
import yaml
from simulation.build_map9_scene import path, move


def footprint_cells(p, footprint, margin, resolution, origin):
    """Conservative cell intersection: inflate polygon by cell circumradius."""
    rot = Rotation.from_euler('z', p[2]).as_matrix()[:2, :2]
    vertices = footprint @ rot.T + p[:2]
    eq = ConvexHull(vertices).equations
    padding = margin + resolution / np.sqrt(2)
    lo = np.floor((vertices.min(axis=0)-padding-origin)/resolution).astype(int)
    hi = np.floor((vertices.max(axis=0)+padding-origin)/resolution).astype(int)
    yy, xx = np.mgrid[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
    xy = np.c_[xx.ravel(), yy.ravel()]*resolution+origin+resolution/2
    inside = np.all(xy @ eq[:, :2].T+eq[:, 2] <= padding+1e-9, axis=1)
    return np.c_[yy.ravel(), xx.ravel()][inside]


def occupied_footprint(p, footprint, margin, r, origin, grid):
    cells = footprint_cells(p, footprint, margin, r, origin)
    valid = ((cells[:, 0] >= 0) & (cells[:, 0] < grid.shape[0])
             & (cells[:, 1] >= 0) & (cells[:, 1] < grid.shape[1]))
    blocked = ~valid
    blocked[valid] = grid[cells[valid, 0], cells[valid, 1]] != 0
    return int(blocked.sum())


class Clearance:
    def __init__(self, model, saved, base_height, cap):
        self.model, self.data, self.saved = model, mujoco.MjData(model), saved
        self.data.qpos[:] = saved['qpos']
        self.base_height, self.cap = base_height, cap
        self.robot = np.flatnonzero(model.geom_group == 0).tolist()
        self.payload = model.geom('payload').id
        self.env = np.array([g for g in range(model.ngeom)
            if model.geom_group[g] in (2, 4) and model.geom(g).name != 'floor'])
        self.set_pose([*saved['initial_base'][:2, 3],
            Rotation.from_matrix(saved['initial_base'][:3, :3]).as_euler('xyz')[2]])
        self.tree = cKDTree(self.data.geom_xpos[self.env])
        self.max_env_radius = float(model.geom_rbound[self.env].max())

    def set_pose(self, p):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler('z', p[2]).as_matrix()
        T[:3, 3] = [*p[:2], self.base_height]
        base = self.model.body('base_link').id
        self.model.body_pos[base] = T[:3, 3]
        self.model.body_quat[base] = Rotation.from_matrix(T[:3, :3]).as_quat(scalar_first=True)
        Tb = T @ self.saved['T_base_payload']
        self.data.mocap_pos[0] = Tb[:3, 3]
        self.data.mocap_quat[0] = Rotation.from_matrix(Tb[:3, :3]).as_quat(scalar_first=True)
        mujoco.mj_forward(self.model, self.data)
        return T

    def check(self, p, voxel_distances=None, include_payload=True):
        self.set_pose(p)
        # Columns: robot/map voxels, robot/furniture, payload/voxels, payload/furniture.
        minima = np.full(4, self.cap)
        worst = [None]*4
        for a in self.robot+([self.payload] if include_payload else []):
            radius = float(self.model.geom_rbound[a])+self.max_env_radius+self.cap
            for j in self.tree.query_ball_point(self.data.geom_xpos[a], radius):
                b = int(self.env[j])
                if np.linalg.norm(self.data.geom_xpos[a]-self.data.geom_xpos[b]) > (
                        self.model.geom_rbound[a]+self.model.geom_rbound[b]+self.cap):
                    continue
                d = mujoco.mj_geomDistance(self.model, self.data, a, b, self.cap, None)
                if voxel_distances is not None and self.model.geom_group[b] == 4 and d < self.cap:
                    voxel_distances[b] = min(voxel_distances.get(b, self.cap), float(d))
                k = (2 if a == self.payload else 0)+(0 if self.model.geom_group[b] == 4 else 1)
                if d < minima[k]:
                    minima[k] = d
                    worst[k] = dict(body=self.model.body(int(self.model.geom_bodyid[a])).name,
                        obstacle=self.model.geom(b).name, signed_distance_m=float(d),
                        obstacle_centre_map_m=self.data.geom_xpos[b].tolist())
        return minima, worst


def interpolate(coords, initial_yaw, final_yaw, linear_step, angular_step):
    samples = [np.r_[coords[0], initial_yaw]]
    for a, b in zip(coords[:-1], coords[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b-a)/linear_step)))
        samples.extend(np.r_[a+(b-a)*u, initial_yaw] for u in np.linspace(0, 1, n+1)[1:])
    delta = (final_yaw-initial_yaw+np.pi) % (2*np.pi)-np.pi
    n = max(1, int(np.ceil(abs(delta)/angular_step)))
    samples.extend(np.r_[coords[-1], initial_yaw+delta*u] for u in np.linspace(0, 1, n+1)[1:])
    return np.array(samples)


def main():
    cfg = yaml.safe_load((ROOT/'simulation/map9.yaml').read_text())
    audit = cfg['offline_audit']; out = path(cfg['output'])
    report = json.loads((out/'report.json').read_text())
    meta = json.loads((path(cfg['snapshot'])/'map.json').read_text())
    r = meta['grid']['resolution_m']; origin = np.array(meta['grid']['origin']['position'][:2])
    grid = np.load(out/'scene_map.npz')['data'].copy()
    points = {p['id']: p for p in meta['guide_points']}
    yy, xx = np.indices(grid.shape)
    xy = np.c_[xx.ravel(), yy.ravel()]*r+origin+r/2
    for b in report['boxes']:
        local = move(np.c_[xy, np.full(len(xy), np.array(b['T'])[2, 3])], np.linalg.inv(b['T']))
        grid.ravel()[np.all(np.abs(local[:, :2]) <= np.array(b['size'])[:2]/2, axis=1)] = 100
    native = yaml.safe_load((out/'native_filtered.yaml').read_text())
    footprint = np.array(native['robot']['footprint'])
    radius = float(np.linalg.norm(footprint, axis=1).max()+audit['chassis_margin_m'])
    start = points[cfg['native_navigation']['start_guide_id']]
    initial_yaw = Rotation.from_quat(start['orientation_xyzw']).as_euler('xyz')[2]
    offsets = footprint_cells(np.array([r/2, r/2, initial_yaw]), footprint,
                              audit['chassis_margin_m'], r, np.zeros(2))
    half = int(np.abs(offsets).max())
    kernel = np.zeros((half*2+1, half*2+1), bool)
    kernel[offsets[:, 0]+half, offsets[:, 1]+half] = True
    valid = binary_erosion(grid == 0, structure=kernel, border_value=0)
    x0, x1, y0, y1 = cfg['work_bounds_xy_m']
    valid &= ((xy[:, 0] > x0) & (xy[:, 0] < x1)
              & (xy[:, 1] > y0) & (xy[:, 1] < y1)).reshape(grid.shape)
    distance = distance_transform_edt(grid == 0)*r-r/np.sqrt(2)
    index = np.full(grid.shape, -1, dtype=np.int32); index[valid] = np.arange(valid.sum())
    rows, cols, costs = [], [], []
    for dy, dx in ((0, 1), (1, 0)):
        a = index[:index.shape[0]-dy or None, :index.shape[1]-dx or None]
        b = index[dy:, dx:]; mask = (a >= 0) & (b >= 0)
        clearance = np.minimum(distance[:a.shape[0], :a.shape[1]], distance[dy:, dx:])[mask]
        cost = r*(1+.5/np.maximum(clearance, .05)**2)
        rows.extend([a[mask], b[mask]]); cols.extend([b[mask], a[mask]]); costs.extend([cost, cost])
    graph = coo_matrix((np.concatenate(costs), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(valid.sum(), valid.sum())).tocsr()
    cells = np.argwhere(valid)
    def cell(point):
        ix, iy = np.floor((np.array(point['position'][:2])-origin)/r).astype(int)
        return iy, ix
    sy, sx = cell(start)
    distances = previous = None
    if index[sy, sx] >= 0:
        distances, previous = dijkstra(graph, directed=True, indices=index[sy, sx], return_predecessors=True)
    model = mujoco.MjModel.from_binary_path(str(out/'scene.mjb'))
    checker = Clearance(model, np.load(out/'state.npz'), cfg['base_height_m'], audit['clearance_cap_m'])
    routes = {}
    for goalid in cfg['native_navigation']['goal_guide_ids']:
        goal = points[goalid]; gy, gx = cell(goal)
        final_yaw = Rotation.from_quat(goal['orientation_xyzw']).as_euler('xyz')[2]
        info = dict(start='G4', goal=f'G{goalid}',
            planner='SciPy Dijkstra, fixed-yaw polygon chassis reference; not native navigation',
            chassis_radius_m=radius, chassis_margin_m=audit['chassis_margin_m'],
            start_clearance_m=float(distance[sy, sx]), goal_clearance_m=float(distance[gy, gx]),
            goal_chassis_blocked_cells=occupied_footprint(np.r_[goal['position'][:2], final_yaw],
                footprint, audit['chassis_margin_m'], r, origin, grid), controller_validated=False)
        target = index[gy, gx]
        if distances is None or target < 0 or not np.isfinite(distances[target]):
            info['status'] = 'no_fixed_yaw_chassis_reference'; routes[str(goalid)] = info
            print(f"G4 -> G{goalid}: {info['status']}", flush=True)
            continue
        chain = [int(target)]
        while chain[-1] != index[sy, sx]: chain.append(int(previous[chain[-1]]))
        coords = cells[chain[::-1]][:, ::-1]*r+origin+r/2
        coords[0] = start['position'][:2]; coords[-1] = goal['position'][:2]
        trajectory = interpolate(coords, initial_yaw, final_yaw,
                                 audit['translation_step_m'], audit['rotation_step_rad'])
        minima, witnesses, grid_hits = [], [], []
        for frame, p in enumerate(trajectory):
            values, worst = checker.check(p)
            minima.append(values)
            grid_hits.append(occupied_footprint(p, footprint, audit['chassis_margin_m'], r, origin, grid))
            for k, w in enumerate(worst):
                if w is not None and values[k] < audit['chassis_margin_m']:
                    witnesses.append(dict(frame=frame, category=k, **w))
            if frame % 200 == 0: print(f"G4 -> G{goalid}: geometry {frame}/{len(trajectory)}", flush=True)
        minima = np.array(minima); grid_hits = np.array(grid_hits)
        collision = minima.min(axis=1) < 0
        margin_fail = minima.min(axis=1) < audit['chassis_margin_m']
        first = np.flatnonzero(collision)
        status = 'reference_geometry_rejected' if collision.any() or grid_hits.any() else (
            'reference_margin_rejected' if margin_fail.any() else 'sampled_reference_clear')
        info.update(status=status, length_m=float(np.linalg.norm(np.diff(coords, axis=0), axis=1).sum()),
            frames=len(trajectory), collision_frames=int(collision.sum()), margin_failure_frames=int(margin_fail.sum()),
            chassis_grid_blocked_frames=int((grid_hits > 0).sum()),
            minimum_geometry_distance_m=float(minima.min()),
            geometry_distance_columns=['robot_voxels', 'robot_furniture', 'payload_voxels', 'payload_furniture'],
            minimum_by_category_m=minima.min(axis=0).tolist(),
            first_collision_frame=int(first[0]) if len(first) else None,
            first_collisions=witnesses[:16], worst_collision=min(witnesses, key=lambda w:w['signed_distance_m']) if witnesses else None,
            continuous_collision_detection=False, translation_step_m=audit['translation_step_m'],
            rotation_step_rad=audit['rotation_step_rad'], minimum_geometry_distance_m_is_capped_at=audit['clearance_cap_m'],
            payload_collision_included=True, unknown_grid_is_blocked=True,
            floor_collision_checked=False, sustained_transport_dynamics_validated=False)
        np.savez_compressed(out/f'route_G4_G{goalid}.npz', poses=trajectory, minimum_distances=minima,
                            chassis_blocked_cells=grid_hits)
        (out/f'route_G4_G{goalid}_collisions.json').write_text(json.dumps(witnesses, ensure_ascii=False, indent=2))
        routes[str(goalid)] = info
        print(f"G4 -> G{goalid}: {status}, {info['collision_frames']} collision frames", flush=True)
    report['route_preview'] = routes
    report['native_navigation_executed'] = False
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(routes, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
