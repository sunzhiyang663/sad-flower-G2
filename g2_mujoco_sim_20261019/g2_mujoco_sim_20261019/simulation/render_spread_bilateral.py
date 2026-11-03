"""Render validated bilateral states and their checked motion in MuJoCo."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from simulation.replay_extraction import configure_graphics
configure_graphics(False)
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from simulation.refine_grasp_elbow_down import load
OUT=ROOT/'simulation/outputs/grasp_spread_bilateral'

def main(output=None):
    OUT=Path(output) if output else ROOT/'simulation/outputs/grasp_spread_bilateral'
    report=json.loads((OUT/'bilateral_final_ik.json').read_text())
    if not report['passed']:raise ValueError('Bilateral geometry checks required')
    _,c,_,_=load(ROOT/'simulation/grasp_wrist_parallel_front.yaml',visuals=True)
    for g in range(c.model.ngeom):
        name=c.model.body(int(c.model.geom_bodyid[g])).name
        if name in c.robot_bodies:
            c.model.geom_rgba[g]=[.72,.77,.81,1]
            if name.startswith('arm_'):c.model.geom_rgba[g]=[.16,.46,.51,1]
            if name.startswith('hand_r_'):c.model.geom_rgba[g]=[.96,.72,.3,1]
        else:
            c.model.geom_group[g]=2
            c.model.geom_rgba[g]=[.47,.54,.60,.23] if name.startswith('scene_') else [.50,.62,.72,1]
    c.model.vis.global_.offwidth=640;c.model.vis.global_.offheight=540
    c.model.vis.quality.offsamples=0;c.model.vis.headlight.ambient[:]=[.55]*3
    renderer=mujoco.Renderer(c.model,540,640);opt=mujoco.MjvOption();opt.geomgroup[:]=[0,1,1,1,1,0]
    font=ImageFont.truetype(str(ROOT/'assets/fonts/NotoSansCJK-Regular.ttc'),25)
    small=ImageFont.truetype(str(ROOT/'assets/fonts/NotoSansCJK-Regular.ttc'),21)
    def frame(state,title):
        c.set_base(np.array(state['base']));c.set_state(state['joints'],np.array(state['payload']))
        height=float(c.data.body('arm_r_link3').xpos[2]-c.data.body('arm_r_link4').xpos[2])
        canvas=Image.new('RGB',(1280,720),'#142738');draw=ImageDraw.Draw(canvas)
        draw.text((18,12),title,font=font,fill='white')
        draw.text((18,52),f'右肘低于肩 {height*100:.1f} cm；腕部偏角 −7.32°；腰部保持原降高姿态，无侧倾或转腰。',font=small,fill='#cce7df')
        for k,angle in enumerate([90,170]):
            cam=mujoco.MjvCamera();cam.lookat[:]=[.55,-.02,.74];cam.distance=1.75;cam.azimuth=angle;cam.elevation=-12
            renderer.update_scene(c.data,cam,scene_option=opt);renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
            pic=Image.fromarray(renderer.render());d=ImageDraw.Draw(pic)
            camera=mujoco.mjv_averageCamera(renderer.scene.camera[0],renderer.scene.camera[1]);right=np.cross(camera.forward,camera.up)
            scale=540*camera.frustum_near/(camera.frustum_top-camera.frustum_bottom)
            def project(p):
                v=p-camera.pos;depth=v@camera.forward
                return (320+scale*(v@right)/depth,270-scale*(v@camera.up)/depth)
            shoulder=project(c.data.body('arm_r_link3').xpos);elbow=project(c.data.body('arm_r_link4').xpos)
            d.line([shoulder,elbow],fill='#ffd87b',width=2)
            for p,col in [(shoulder,'#59fa75'),(elbow,'#ff634a')]:d.ellipse([p[0]-6,p[1]-6,p[0]+6,p[1]+6],fill=col)
            canvas.paste(pic,(640*k,95))
        dx,dy=report['base_shift_xy_m']
        draw.text((18,651),f'抓取站位相对原方案：前移 {dx*100:.1f} cm，左移 {dy*100:.1f} cm；绿点肩 / 红点肘。',font=small,fill='white')
        draw.text((18,684),'MuJoCo 运动学与碰撞模型回放；未验证真实摩擦、握力或负载。',font=small,fill='#ffd38a')
        return canvas
    try:
        for key,title in [('right_grasp','右手抓取，左手在货架外等待'),('paired_grasp','右手拽出 19 cm 后，左手接入托抓')]:
            if key=='paired_grasp' and report.get('left_forward_shift_m'):
                title+=f"（托点前移 {100*report['left_forward_shift_m']:g} cm，左腕下调 {-1000*report.get('left_vertical_shift_m',0):g} mm）"
            frame(report[key],title).save(OUT/f'{key}_fullbody.png')
        path=OUT/'motion.json'
        if path.exists():
            motion=json.loads(path.read_text())
            if not motion['passed']:raise ValueError('Motion checks required')
            states=motion['states']
            process=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','12','-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'spread_bilateral_grasp.mp4')],stdin=subprocess.PIPE)
            try:
                for i in np.linspace(0,len(states)-1,180).astype(int):process.stdin.write(frame(states[i],states[i]['label']).tobytes())
            finally:
                process.stdin.close()
                if process.wait()!=0:raise RuntimeError('ffmpeg failed')
    finally:renderer.close()
    print(OUT,flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output');args=parser.parse_args();main(args.output)
