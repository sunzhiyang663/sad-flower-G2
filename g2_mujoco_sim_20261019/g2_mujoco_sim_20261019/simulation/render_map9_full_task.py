"""Record the seven-step offline transfer directly in MuJoCo, as MP4."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from simulation.replay_extraction import configure_graphics,STAGE_NAMES
configure_graphics(False)
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation
import yaml
from simulation.full_task_config import load_task
from simulation.build_map9_scene import path,pose
from simulation.map9_route_preview import Clearance

NAMES={**STAGE_NAMES,'default':'默认全身关节姿态','prepare_hands':'手指准备',
    'base_retreat':'放置后底盘直退，壳体留在台面','reset_hands':'退离台面后手指复位','reset_arms':'双臂依次复位','reset_body':'腰颈恢复默认姿态',
    'carry_backoff':'双手持壳直退，离开货架后再转弯','paired_table_approach':'降低至桌沿高度，底盘直进送入自由前端',
    'right_arm_push':'右手沿台面推送，减少底盘向桌下伸入',
    'right_partial_drag':'右手先部分拖出，壳体仍搭在货架上','paired_full_drag':'左手已包握，双手共同退出货架',
    'paired_table_hover':'双手保持包握，将自由前端送到桌沿上方','front_land':'双手共同下降，自由前端先落到台面',
    'right_push_on_table':'左手已撤离，右手沿台面向前推送',
    'arm_plan':'远处完成双臂预摆','base_approach':'保持双臂构型，底盘直进靠近','base_pull':'保持右手抓取，底盘直退抽出','left_entry':'左手从侧下方进入重心下方','left_lower_entry':'左手调整到壳体底面下方的接货高度','table_hover':'右手将壳体送至台面上方',
    'left_release':'左手松开托底包握','left_drop':'左手稍降，解除托底接触',
    'left_withdraw':'左手退出台面下方','place_on_table':'右手下降，壳体落到台面',
    'left_park':'左手收回身体侧方，让出前推空间',
    'right_open_clear':'右手撤离后完全张开','right_release':'右手张开，解除负载附着','right_withdraw':'右手撤回，壳体留在台面'}


def make_sequence(goal,cfg,task,out,meta):
    pickup=np.load(out/'pickup.npz');place=np.load(out/'placement.npz')
    empty=np.load(out/f'route_G{task["start_guide_id"]}_G4.npz');loaded=np.load(out/f'route_G4_G{goal}.npz')
    points={p['id']:p for p in meta['guide_points']};G4=pose(points[4],cfg['base_height_m']);Gdest=pose(points[goal],cfg['base_height_m'])
    base_offsets=pickup['base_offsets'] if 'base_offsets' in pickup else np.repeat(np.eye(4)[None],len(pickup['qpos']),axis=0)
    shell_world=G4@base_offsets[0]@pickup['payload'][0]
    fps=task['recording']['fps'];timeline=[];timestamps=[];duration=0.
    def put(p,q,T,label,seconds,attached=False):
        nonlocal duration
        timestamps.append(duration);timeline.append((p.copy(),q.copy(),T.copy(),label,attached));duration+=seconds
    def nav(record,q,label,attached):
        poses=record['poses'];longitudinal=record['longitudinal'];last_sign=0
        for k,p in enumerate(poses):
            sign=1 if k==len(poses)-1 or longitudinal[k]>=0 else -1
            speed=task['vehicle']['speed_m_s'] if sign>0 else task['vehicle']['reverse_speed_m_s']
            seconds=(max(float(record['ds'][k])/speed,abs(float(record['dyaw'][k]))/task['vehicle']['max_yaw_speed_rad_s']) if k<len(poses)-1 else 1/fps)
            if last_sign and sign!=last_sign:seconds+=task['vehicle']['switch_gear_hold_s']
            last_sign=sign
            Tb=np.eye(4);Tb[:3,:3]=Rotation.from_euler('z',p[2]).as_matrix();Tb[:3,3]=[*p[:2],cfg['base_height_m']]
            T=pickup['payload'][-1] if attached else np.linalg.inv(Tb)@shell_world
            put(p,q,T,label+(' · 前进' if sign>0 else ' · 倒车'),seconds,attached)
    start=empty['poses'][0]
    Tb=np.eye(4);Tb[:3,:3]=Rotation.from_euler('z',start[2]).as_matrix();Tb[:3,3]=[*start[:2],cfg['base_height_m']]
    put(start,pickup['default_qpos'],np.linalg.inv(Tb)@shell_world,'1 默认姿态，从 G6 出发',1.)
    nav(empty,pickup['default_qpos'],'1 空载导航到货架 G4',False)
    pg4=empty['poses'][-1]
    put(pg4,pickup['qpos'][0],pickup['payload'][0],'2 识别壳体位姿（离线假定观测）',2.)
    for k,label in enumerate(pickup['labels']):
        if label=='default':continue
        seconds=1/float(np.load(path(cfg['carry_reference'])/'replay.npz')['fps'])
        if label in ('lower_waist','prepare_hands','arm_plan'):
            seconds=max(task['motion'].get('minimum_sample_duration_s',1/fps),float(np.abs(pickup['qpos'][k]-pickup['qpos'][k-1]).max())/task['motion']['joint_speed_rad_s'])
        number='3' if label=='lower_waist' else '4' if label in ('prepare_hands','arm_plan','left_open','left_support_front','left_support_wait','right_open','right_pregrasp','right_approach') else '5'
        Tb=G4@base_offsets[k];p=np.r_[Tb[:2,3],Rotation.from_matrix(Tb[:3,:3]).as_euler('xyz')[2]]
        if label in ('base_approach','base_pull','right_partial_drag','paired_full_drag','carry_backoff'):
            speed=task['vehicle']['speed_m_s'] if label=='base_approach' else task['vehicle']['reverse_speed_m_s']
            seconds=max(task['motion']['minimum_sample_duration_s'],float(np.linalg.norm(base_offsets[k,:3,3]-base_offsets[k-1,:3,3]))/speed)
        put(p,pickup['qpos'][k],pickup['payload'][k],number+' '+NAMES.get(str(label),str(label)),seconds,
            label in ('partial_pull','base_pull','right_partial_drag','paired_full_drag','left_lower_entry','left_entry','verify_pose','left_seat','left_grasp','full_pull','lift','carry_backoff'))
    nav(loaded,pickup['qpos'][-1],f'6 携壳导航到 G{goal}',True)
    pg=loaded['poses'][-1]
    for k,label in enumerate(place['labels']):
        if 'base_offsets' in place:
            Tb=Gdest@place['base_offsets'][k];pg=np.r_[Tb[:2,3],Rotation.from_matrix(Tb[:3,:3]).as_euler('xyz')[2]]
        seconds=max(task['motion'].get('minimum_sample_duration_s',1/fps),float(np.abs(place['qpos'][k]-place['qpos'][max(0,k-1)]).max())/task['motion']['joint_speed_rad_s'])
        if label in ('paired_table_approach','right_push_on_table','base_retreat') and k:
            seconds=max(seconds,float(np.linalg.norm(place['base_offsets'][k,:3,3]-place['base_offsets'][k-1,:3,3]))/task['vehicle']['reverse_speed_m_s'])
        put(pg,place['qpos'][k],place['payload'][k],'7 '+NAMES.get(str(label),str(label)),seconds,
            label not in ('right_release','right_withdraw','right_open_clear','base_retreat','reset_hands','reset_arms','reset_body'))
    put(pg,place['qpos'][-1],place['payload'][-1],'7 放置、退离和复位完成' if task.get('spread_source') else '7 放置完成 · 壳体底面高 75 cm',2.)
    return timeline,np.array(timestamps),duration


def main(goal,still=False):
    cfg=yaml.safe_load((ROOT/'simulation/map9.yaml').read_text());task=load_task();out=path(task['output'])
    validation=json.loads((out/'validation.json').read_text())
    if not validation['passed']:raise RuntimeError('Cannot render a successful task without validation')
    meta=json.loads((path(cfg['snapshot'])/'map.json').read_text())
    sequence,times,duration=make_sequence(goal,cfg,task,out,meta)
    model=mujoco.MjModel.from_binary_path(str(out/'scene.mjb'));saved=dict(np.load(out/'state.npz'))
    checker=Clearance(model,saved,cfg['base_height_m'],.1)
    width,height=task['recording']['video_size'];fps=task['recording']['fps']
    model.vis.global_.offwidth=width;model.vis.global_.offheight=height
    if 'msaa_samples' in task['recording']:model.vis.quality.offsamples=task['recording']['msaa_samples']
    model.vis.headlight.ambient[:]=[.5,.5,.5]
    model.vis.headlight.diffuse[:]=[.7,.7,.7]
    model.vis.headlight.specular[:]=[.1,.1,.1]
    renderer=mujoco.Renderer(model,height,width,max_geom=model.ngeom+100)
    option=mujoco.MjvOption();option.geomgroup[:]=[0,1,1,1,1,0]
    camera=mujoco.MjvCamera();camera.distance=3.2;camera.elevation=-24
    font_path=str(ROOT/'assets/fonts/NotoSansCJK-Regular.ttc')
    font=ImageFont.truetype(font_path,max(16,round(22*width/960)))
    small=ImageFont.truetype(font_path,max(12,round(16*width/960)))
    grid=np.load(path(cfg['output'])/'scene_map.npz')['data']
    origin=np.array(meta['grid']['origin']['position'][:2]);resolution=meta['grid']['resolution_m']
    x0,x1,y0,y1=cfg['work_bounds_xy_m'];ix0,iy0=np.floor((np.array([x0,y0])-origin)/resolution).astype(int)
    ix1,iy1=np.ceil((np.array([x1,y1])-origin)/resolution).astype(int)
    crop=grid[iy0:iy1,ix0:ix1][::-1];rgb=np.full((*crop.shape,3),220,dtype=np.uint8)
    rgb[crop==0]=[247,250,252];rgb[crop==100]=[44,65,81]
    mini=Image.fromarray(rgb).resize((190,170),Image.NEAREST)
    def pixel(p):return ((p[0]-x0)/(x1-x0)*190,(y1-p[1])/(y1-y0)*170)
    md=ImageDraw.Draw(mini)
    for filename in (f'route_G{task["start_guide_id"]}_G4.npz',f'route_G4_G{goal}.npz'):
        route=np.load(out/filename)['poses'];md.line([pixel(p) for p in route],fill='#129c8c',width=2)
    for guide in meta['guide_points']:
        if guide['id'] in (task['start_guide_id'],4,goal):
            x,y=pixel(guide['position']);md.text((x+3,y-14),'G'+str(guide['id']),font=small,fill='#693ea3')
    process=None;count=int(np.ceil(duration*fps))
    if not still:
        process=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(out/f'G6_G4_G{goal}_full_task.mp4')],stdin=subprocess.PIPE)
    frameids=range(count) if not still else np.linspace(0,count-1,9).astype(int)
    previous=None;pixels=None;chapters=[];previous_label=None
    try:
        for frame in frameids:
            i=min(len(sequence)-1,max(0,int(np.searchsorted(times,frame/fps,side='right')-1)))
            p,q,T,label,attached=sequence[i]
            if i==previous and not still:process.stdin.write(pixels);continue
            checker.data.qpos[:]=q;checker.saved['T_base_payload']=T;checker.set_pose(p)
            camera.lookat[:]=[p[0]+.50*np.cos(p[2]),p[1]+.50*np.sin(p[2]),.82]
            camera.azimuth=np.rad2deg(p[2])+85
            # The robot reaches G4 facing the rack. The extraction camera keeps
            # that side view for the complete manipulation, including placement.
            renderer.update_scene(checker.data,camera,scene_option=option)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
            im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
            draw.rectangle((0,0,width,68),fill='#142738')
            draw.text((15,6),f'地图 9 · G6 → G4 → G{goal} | {label}',font=font,fill='white')
            cross=task['motion'].get('preposition',{}).get('mode')=='cross_wrap'
            summary='远处依次预摆 → 右手部分拖出 → 左手包握、双手抬起 → 前端落台、松左手、右手推入' if cross else '默认姿态 → 识别 → 远处预摆 → 直进抓取、直退托底 → 持壳通行 → 放置'
            draw.text((15,37),summary,font=small,fill='#d8e7ef')
            panel=mini.copy();pd=ImageDraw.Draw(panel);x,y=pixel(p)
            pd.ellipse((x-4,y-4,x+4,y+4),fill='#e2573f')
            pd.line((x,y,x+13*np.cos(p[2]),y-13*np.sin(p[2])),fill='#e2573f',width=2)
            im.paste(panel,(width-200,78));draw=ImageDraw.Draw(im)
            draw.rectangle((0,height-54,width,height),fill='#142738')
            draw.text((15,height-50),'离线运动学预演 · 假定视觉观测 · OMPL 车辆路径 · 未接原厂闭环',font=small,fill='#d8e7ef')
            note='壳体从模拟感知点云滤除，碰撞几何保留；抓放使用运动学附着辅助'
            if cross and label.startswith('7 '):
                note='自由前端先接触桌面，左手随后松开撤出，右手沿台面推入；仅运动学预演'
            elif task['motion'].get('preposition') and label.startswith('7 '):
                note=f"放置前端悬伸 {max(0.,-task['motion']['placement_edge_gap_m'])*100:g} cm，几何重心仍在台面内；未验证负载承载能力"
            draw.text((15,height-27),note,font=small,fill='#ffd382')
            pixels=im.tobytes();previous=i
            if label!=previous_label:
                chapters.append(dict(time_s=frame/fps,label=label));previous_label=label
            if process:process.stdin.write(pixels)
            if still:im.save(out/f'G{goal}_preview_{frame:04d}.png')
            if label.startswith('2 '):im.save(out/f'G{goal}_detection.png')
            if i==len(sequence)-1:im.save(out/f'G{goal}_placed.png')
            if still or frame%(fps*5)==0:print(f'G{goal} frame {frame}/{count}: {label}',flush=True)
    finally:
        renderer.close()
        if process:
            process.stdin.close()
            if process.wait()!=0:raise RuntimeError('ffmpeg failed')
    if not still:
        (out/f'G{goal}_video.json').write_text(json.dumps(dict(frames=count,fps=fps,duration_s=count/fps,chapters=chapters,
            seven_steps=True,placement_completed=previous==len(sequence)-1,hardware_commands_sent=False),ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--goal',type=int,choices=(1,5),required=True);p.add_argument('--still',action='store_true');a=p.parse_args();main(a.goal,a.still)
