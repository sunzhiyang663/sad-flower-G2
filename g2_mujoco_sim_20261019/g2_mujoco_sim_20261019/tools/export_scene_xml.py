"""Export editable current MJCF with relative mesh assets; no robot IO."""
from pathlib import Path
import os,sys,tempfile,json
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
import yaml,numpy as np,mujoco
from simulation.build_map9_scene import build
from simulation.full_task_config import load_task

def main():
    cfg=yaml.safe_load((ROOT/'simulation/map9.yaml').read_text())
    with tempfile.TemporaryDirectory(prefix='g2_scene_export_') as tmp:
        cfg['output']=tmp;p=Path(tmp)/'config.yaml';p.write_text(yaml.safe_dump(cfg))
        baseline,spec=build(p,export_only=True)
        mujoco.MjModel.from_xml_string(spec.to_xml())
        current=mujoco.MjModel.from_binary_path(str(ROOT/load_task()['output']/'scene.mjb'))
        assert baseline.ngeom==current.ngeom and baseline.nbody==current.nbody
        assert baseline.names==current.names, 'Scene topology changed; regenerate a matching scene first'
        out=ROOT/'scenes/map9_current.xml'
        # The last compiled spec supplies mesh assets; the current model supplies
        # geometry and collision masks, including the accepted cloud cleanup.
        mujoco.mj_saveLastXML(str(out),current)
        tree=ET.parse(out)
        for elem in tree.getroot().iter():
            if elem.tag in ('mesh','texture','hfield') and 'file' in elem.attrib:
                path=Path(elem.get('file'))
                if not path.is_absolute():path=ROOT/path
                assert path.is_file(),path
                elem.set('file',os.path.relpath(path,out.parent))
        # mj_saveLastXML does not transfer every runtime visual/collision flag.
        for elem in tree.getroot().iter('geom'):
            name=elem.get('name')
            if name:
                gid=current.geom(name).id
                for attr in ('group','contype','conaffinity'):
                    elem.set(attr,str(int(getattr(current,'geom_'+attr)[gid])))
                elem.set('rgba',' '.join(map(str,current.geom_rgba[gid])))
        for elem in tree.getroot().iter('body'):
            name=elem.get('name')
            if name:
                bid=current.body(name).id
                elem.set('pos',' '.join(map(str,current.body_pos[bid])))
                for attr in ('euler','axisangle','xyaxes','zaxis'):elem.attrib.pop(attr,None)
                elem.set('quat',' '.join(map(str,current.body_quat[bid])))
        visual=tree.getroot().find('visual')
        if visual is None:visual=ET.SubElement(tree.getroot(),'visual')
        global_=visual.find('global')
        if global_ is None:global_=ET.SubElement(visual,'global')
        global_.set('offwidth',str(current.vis.global_.offwidth))
        global_.set('offheight',str(current.vis.global_.offheight))
        quality=visual.find('quality')
        if quality is None:quality=ET.SubElement(visual,'quality')
        quality.set('offsamples',str(current.vis.quality.offsamples))
        tree.write(out,encoding='utf-8',xml_declaration=True)
        rebuilt=mujoco.MjModel.from_xml_path(str(out))
        errors={}
        for field in ('geom_type','geom_size','geom_pos','geom_quat','geom_bodyid','geom_group','geom_contype','geom_conaffinity','body_pos','body_quat','jnt_range'):
            a,b=getattr(current,field),getattr(rebuilt,field)
            assert a.shape==b.shape,(field,a.shape,b.shape)
            error=float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.size else 0.
            errors[field]=error
            assert np.allclose(a,b,rtol=1e-5,atol=2e-6),(field,error)
        (ROOT/'scenes/export_check.json').write_text(json.dumps({'passed':True,'nq':rebuilt.nq,'ngeom':rebuilt.ngeom,'maximum_absolute_differences':errors,'note':'MJCF numeric formatting can round floats. Recorded MJB and trajectories remain the reference.'},indent=2))
        print(out,flush=True)
if __name__=='__main__':main()
