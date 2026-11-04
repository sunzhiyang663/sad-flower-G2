"""Select an offline transfer profile without replacing historical profiles."""
import os
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
def load_task(filename=None):
    filename=Path(filename or os.environ.get('G2_SIM_TASK','simulation/map9_spread_forward5_full.yaml'))
    def merge(base,overrides):
        result=dict(base)
        for key,value in overrides.items():
            result[key]=merge(result[key],value) if isinstance(value,dict) and isinstance(result.get(key),dict) else value
        return result
    def read(path,seen):
        path=path.resolve()
        if path in seen:raise ValueError('Cyclic task configuration inheritance')
        value=yaml.safe_load(path.read_text(encoding='utf-8'))
        parent=value.pop('extends',None)
        return merge(read(ROOT/parent,seen|{path}),value) if parent else value
    return read(ROOT/filename,set())
