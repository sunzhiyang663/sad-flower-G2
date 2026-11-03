"""MuJoCo close-up MP4 of the validated drag, wrap and front-first placement."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from simulation.replay_extraction import configure_graphics
configure_graphics(False)
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import yaml
from simulation.full_task_config import load_task
from simulation.build_map9_scene import path
from simulation.map9_route_preview import Clearance
from simulation.render_map9_full_task import make_sequence


def main(still=False):
    cfg=yaml.safe_load((ROOT/'simulation/map9.yaml').read_text());task=load_task();out=path(task['output'])
    if not json.loads((out/'validation.json').read_text())['passed']:raise RuntimeError('Full sequence validation required')
    meta=json.loads((path(cfg['snapshot'])/'map.json').read_text());sequence,times,duration=make_sequence(1,cfg,task,out,meta)
    indices=[k for k,s in enumerate(sequence) if s[3].startswith(('5 ','7 '))]
    spans=np.diff(np.r_[times,duration]);starts=np.r_[0.,np.cumsum(spans[indices])[:-1]];length=float(spans[indices].sum())
    model=mujoco.MjModel.from_binary_path(str(out/'scene.mjb'));saved=dict(np.load(out/'state.npz'));checker=Clearance(model,saved,cfg['base_height_m'],.1)
    width,height=960,540;view_w,view_h=480,420;fps=8
    model.vis.global_.offwidth=view_w;model.vis.global_.offheight=view_h
    if 'msaa_samples' in task['recording']:model.vis.quality.offsamples=task['recording']['msaa_samples']
    model.vis.headlight.ambient[:]=[.5,.5,.5];model.vis.headlight.diffuse[:]=[.7,.7,.7]
    renderer=mujoco.Renderer(model,view_h,view_w,max_geom=model.ngeom+100)
    option=mujoco.MjvOption();option.geomgroup[:]=[0,1,1,1,1,0]
    camera=mujoco.MjvCamera();camera.distance=1.05;camera.elevation=-12
    fontpath=str(ROOT/'assets/fonts/NotoSansCJK-Regular.ttc');font=ImageFont.truetype(fontpath,23);small=ImageFont.truetype(fontpath,18)
    def render(k,angles):
        p,q,T,label,_=sequence[k];checker.data.qpos[:]=q;checker.saved['T_base_payload']=T;Tb=checker.set_pose(p);world=Tb@T
        camera.lookat[:]=world[:3,3]+[0,0,-.015]
        views=[]
        for angle in angles:
            camera.azimuth=np.rad2deg(p[2])+angle;renderer.update_scene(checker.data,camera,scene_option=option);renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
            views.append(Image.fromarray(renderer.render()))
        return views,label
    process=None
    try:
        if still:
            k=next(k for k in indices if '双手抬升' in sequence[k][3])
            views,label=render(k,[65,-65,135,-135]);sheet=Image.new('RGB',(960,900),'#142738');draw=ImageDraw.Draw(sheet)
            for n,v in enumerate(views):
                x=(n%2)*480;y=(n//2)*450;sheet.paste(v,(x,y+30));draw.text((x+10,y),str([65,-65,135,-135][n]),font=font,fill='white')
            sheet.save(out/'closeup_angles.png');return
        process=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'grasp_front_land_closeup.mp4')],stdin=subprocess.PIPE)
        previous=None;pixels=None;chapters=[];last_label=None;count=int(np.ceil(length*fps))
        for frame in range(count):
            n=min(len(indices)-1,max(0,int(np.searchsorted(starts,frame/fps,side='right')-1)));k=indices[n]
            if k!=previous:
                views,label=render(k,[65,-65]);canvas=Image.new('RGB',(width,height),'#142738');canvas.paste(views[0],(0,70));canvas.paste(views[1],(480,70));draw=ImageDraw.Draw(canvas)
                draw.text((15,6),'抓放近景 · '+label,font=font,fill='white');draw.text((15,39),'右手部分拖出 → 左手横向包握 → 双手抬起 → 前端落台 → 松左手、右手推入',font=small,fill='#d8e7ef')
                draw.text((15,502),'同一条已验证轨迹的两个视角 · 离线运动学附着，未验证真实负载与摩擦',font=small,fill='#ffd382')
                pixels=canvas.tobytes();previous=k
                if label!=last_label:chapters.append(dict(time_s=frame/fps,label=label));last_label=label
            process.stdin.write(pixels)
            if frame%(fps*5)==0:print(f'Close-up {frame}/{count}',flush=True)
        (out/'closeup_video.json').write_text(json.dumps(dict(frames=count,fps=fps,duration_s=count/fps,chapters=chapters,source='validated G1 full sequence',hardware_commands_sent=False),ensure_ascii=False,indent=2))
    finally:
        renderer.close()
        if process:
            process.stdin.close()
            if process.wait()!=0:raise RuntimeError('ffmpeg failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--still',action='store_true');args=parser.parse_args();main(args.still)
