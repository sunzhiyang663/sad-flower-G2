"""Save plain FM diagnostic plots on a headless server (Matplotlib Agg)."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',help='Training directory containing metrics.jsonl')
    parser.add_argument('--trajectory',help='Generated candidate NPZ')
    parser.add_argument('--reference',help='Expert episode NPZ')
    parser.add_argument('--window-start',type=int,default=0)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if not args.run and not args.trajectory:parser.error('Provide --run or --trajectory')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=ROOT/args.output;output.mkdir(parents=True,exist_ok=True)
    if args.run:
        rows=[json.loads(line) for line in (ROOT/args.run/'metrics.jsonl').read_text().splitlines() if line.strip()]
        fig,ax=plt.subplots(figsize=(8,4))
        for key in ('loss','train_probe','validation'):
            points=[(row['step'],row[key]) for row in rows if key in row]
            if points:ax.plot(*zip(*points),label=key)
        ax.set(xlabel='Training step',ylabel='CFM loss');ax.legend();ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(output/'loss.png',dpi=160);plt.close(fig)
    if args.trajectory:
        with np.load(ROOT/args.trajectory,allow_pickle=False) as record:
            q,u,t=record['states'].copy(),record['actions'].copy(),record['timestamps'].copy()
        if u.shape!=(len(q)-1,3) or t.shape!=(len(q),) or np.any(np.diff(t)<=0):
            raise ValueError('Malformed timed state/action trajectory')
        fig,ax=plt.subplots(figsize=(7,6))
        ax.plot(q[:,0],q[:,1],label='FM')
        ax.scatter(q[[0,-1],0],q[[0,-1],1],marker='x',label='Endpoints')
        if args.reference:
            with np.load(ROOT/args.reference) as record:
                ref=record['states'][args.window_start:args.window_start+len(q)]
            if len(ref)!=len(q):raise ValueError('Reference window is too short')
            ax.plot(ref[:,0],ref[:,1],'--',label='Expert')
        ax.set(xlabel='Map x (m)',ylabel='Map y (m)',title='Trajectory comparison (obstacles not shown)')
        ax.axis('equal');ax.legend();fig.tight_layout();fig.savefig(output/'trajectory.png',dpi=160);plt.close(fig)
        dt=np.diff(t);theta=q[:-1,2]
        world=np.column_stack([np.cos(theta)*u[:,0]-np.sin(theta)*u[:,1],np.sin(theta)*u[:,0]+np.cos(theta)*u[:,1],u[:,2]])
        residual=np.diff(q,axis=0)-dt[:,None]*world
        fig,axes=plt.subplots(3,1,figsize=(9,9),sharex=True)
        axes[0].plot(t[:-1],u[:,0],label='vx');axes[0].plot(t[:-1],u[:,1],label='vy')
        axes[0].plot(t[:-1],np.linalg.norm(u[:,:2],axis=1),label='speed');axes[0].set_ylabel('m/s')
        axes[1].plot(t[:-1],u[:,2],label='omega');axes[1].set_ylabel('rad/s')
        axes[2].plot(t[:-1],np.linalg.norm(residual[:,:2],axis=1),label='position residual (m)')
        axes[2].plot(t[:-1],np.abs(residual[:,2]),label='heading residual (rad)')
        axes[2].set_xlabel('Physical time (s)')
        for ax in axes:ax.legend();ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(output/'controls_residual.png',dpi=160);plt.close(fig)
    print(output)


if __name__=='__main__':main()
