"""Seeded bounded wrist IK with collision and joint-limit avoidance, offline only."""
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

def translate(c,joints,battery,base,goals,contacts,attachment=None,step_m=.004):
    c.set_base(base);c.set_state(joints,battery)
    names=[n for s in goals for n in c.names[s]];ids=np.array([int(c.model.joint(n).qposadr[0]) for n in names]);lo,hi=np.array([c.model.joint(n).range for n in names]).T
    starts={s:c.c.get_transform_frame_to_world(f'arm_{"l" if s=="left" else "r"}_end_link','body').as_matrix().copy() for s in goals}
    count=max(1,int(np.ceil(max(np.linalg.norm(T[:3,3]-starts[s][:3,3]) for s,T in goals.items())/step_m)))
    ref=c.data.qpos.copy();seed=ref[ids].copy();path=[]
    moving={g for g in c.robot if any(c.model.body(int(c.model.geom_bodyid[g])).name.startswith((f'arm_{"l" if s=="left" else "r"}_',f'hand_{"l" if s=="left" else "r"}_')) for s in goals)}
    pairs=np.array([(a,b) for a,b in c.collision_pairs if a!=c.payload and b!=c.payload and (a in moving or b in moving)])
    aa,bb=pairs.T;radii=c.model.geom_rbound[aa]+c.model.geom_rbound[bb]+.01
    for k,f in enumerate(np.linspace(0,1,count+1)[1:]):
        targets={s:T.copy() for s,T in starts.items()}
        for s in targets:targets[s][:3,3]=(1-f)*starts[s][:3,3]+f*goals[s][:3,3]
        def residual(x):
            q=ref.copy();q[ids]=x;c.c.update(q);err=[]
            for s,T in targets.items():
                actual=c.c.get_transform_frame_to_world(f'arm_{"l" if s=="left" else "r"}_end_link','body').as_matrix()
                err.extend(1500*(actual[:3,3]-T[:3,3]));err.extend(400*Rotation.from_matrix(T[:3,:3].T@actual[:3,:3]).as_rotvec())
            d=np.zeros(len(pairs));near=np.linalg.norm(c.data.geom_xpos[aa]-c.data.geom_xpos[bb],axis=1)<radii
            for i in np.flatnonzero(near):d[i]=max(0,.0035-mujoco.mj_geomDistance(c.model,c.data,int(aa[i]),int(bb[i]),.005,None))
            return np.r_[err,10000*d,.008*(x-seed),.3*np.maximum(.10-np.minimum(x-lo,hi-x),0)]
        fit=least_squares(residual,np.clip(seed,lo+1e-9,hi-1e-9),bounds=(lo,hi),max_nfev=180,ftol=1e-8,xtol=1e-8,gtol=1e-8)
        e=residual(fit.x);pose_errors=e[:6*len(goals)].reshape(-1,6)
        if np.max(np.linalg.norm(pose_errors[:,:3],axis=1))/1500>.00015 or np.max(np.linalg.norm(pose_errors[:,3:],axis=1))/400>.0005:
            raise ValueError(f'Spread Cartesian IK failed {k}/{count}: {pose_errors.tolist()}')
        seed=fit.x.copy();actual=c.current_joints()
        if attachment is not None:battery=c.c.get_transform_frame_to_world('arm_r_end_link','body').as_matrix()@attachment
        c.set_state(actual,battery);c.collisions(contacts);path.append(dict(joints=actual))
        if (k+1)%10==0 or k+1==count:print(f'  Cartesian samples {k+1}/{count}',flush=True)
    return dict(path=path,final_battery=np.asarray(battery).tolist())
