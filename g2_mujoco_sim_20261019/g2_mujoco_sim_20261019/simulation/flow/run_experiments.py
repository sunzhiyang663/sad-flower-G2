"""Sequential, resumable multi-seed FM experiments; subprocesses use no shell."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def flags(options,allowed):
    if set(options)-set(allowed):raise ValueError(f'Unsupported experiment options: {set(options)-set(allowed)}')
    result=[]
    for key,value in options.items():result.extend(['--'+key.replace('_','-'),str(value)])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    config_path=ROOT/args.config
    config=json.loads(config_path.read_text(encoding='utf-8-sig'))
    seeds=config['seeds'];splits=config.get('splits',['validation','test'])
    if not seeds or len(seeds)!=len(set(seeds)) or any(type(s)!=int or s<0 for s in seeds):raise ValueError('Seeds must be distinct nonnegative integers')
    if not splits or len(splits)!=len(set(splits)) or any(s not in ('train','validation','test') for s in splits):raise ValueError('Invalid splits')
    train_flags=flags(config.get('train',{}),('steps','batch_size','horizon','stride','width','depth','heads','lr','log_every','ode_steps','overfit_windows'))
    eval_flags=flags(config.get('evaluate',{}),('max_windows','stride','candidates','steps'))
    device=config.get('device','auto')
    output=ROOT/args.output
    data=ROOT/config['data']
    manifest=data/'dataset_manifest.json'
    if not manifest.exists():manifest=data/'manifest.json'
    signature=dict(config_sha256=sha(config_path),manifest_sha256=sha(manifest),
        code_sha256={name:sha(HERE/name) for name in ('train_g2.py','evaluate_fm.py','dataset.py','model.py','flow_matcher.py','fm_inference.py','run_experiments.py','plot_results.py','summarize_results.py')})
    if args.dry_run:
        print(json.dumps(dict(config=config,output=str(output),signature=signature),indent=2));return
    output.mkdir(parents=True,exist_ok=True)
    status_path=output/'experiments.json'
    if args.resume:
        status=json.loads(status_path.read_text())
        if status['signature']!=signature:raise ValueError('Configuration, code or dataset changed; choose a new output')
    else:
        if any(output.iterdir()):raise ValueError('Output not empty; use --resume or a new directory')
        status=dict(signature=signature,config=config,seeds={},complete=False)
    lock=output/'.experiment.lock'
    descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(descriptor,str(os.getpid()).encode());os.close(descriptor)
    def persist():
        temp=output/'experiments.tmp';temp.write_text(json.dumps(status,indent=2),encoding='utf-8');temp.replace(status_path)
    env={**os.environ,'OMP_NUM_THREADS':str(config.get('cpu_threads',4))}
    def stage(record,name,script,arguments,required):
        history=record.setdefault(name,[])
        if history and history[-1]['status']=='completed':
            item=history[-1]
            if any(not (output/file).is_file() or sha(output/file)!=digest for file,digest in item['artifacts'].items()):
                raise ValueError(f'Completed stage artifacts missing/changed: {name}')
            return output/item['directory']
        directory=Path(record['directory'])/name/f'attempt_{len(history)+1:03d}'
        target=output/directory
        command=[sys.executable,str(HERE/script),*arguments,'--output',str(target)]
        log=output/directory.parent/f'attempt_{len(history)+1:03d}.log'
        log.parent.mkdir(parents=True,exist_ok=True)
        item=dict(status='running',directory=directory.as_posix(),command=command,log=str(log.relative_to(output)))
        history.append(item);persist();clock=time.monotonic()
        try:
            with log.open('w',encoding='utf-8') as stream:
                result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            item['returncode']=result.returncode
            artifacts=[target/name for name in required]
            if result.returncode or not all(p.is_file() for p in artifacts):
                item['status']='failed';return None
            item.update(status='completed',artifacts={p.relative_to(output).as_posix():sha(p) for p in artifacts})
            return target
        except BaseException as exc:
            item.update(status='interrupted',reason=str(exc));raise
        finally:
            item['elapsed_seconds']=time.monotonic()-clock;persist()
    try:
        persist();success=True
        for seed in seeds:
            record=status['seeds'].setdefault(str(seed),dict(directory=f'seed_{seed}'))
            trained=stage(record,'train','train_g2.py',['--data',str(data),'--device',device,'--seed',str(seed),*train_flags],['latest.pt','best.pt','config.json','normalizer.json'])
            if trained is None:success=False;continue
            for split in splits:
                evaluated=stage(record,'eval_'+split,'evaluate_fm.py',['--checkpoint',str(trained/'best.pt'),
                    '--data',str(data),'--split',split,'--device',device,'--seed',str(seed),*eval_flags],['evaluation.json'])
                if evaluated is None:success=False
            if config.get('plots',False):
                if stage(record,'plots','plot_results.py',['--run',str(trained)],['loss.png']) is None:success=False
        # Summary is regenerated from successful reports; failed stages stay in experiments.json.
        summary_command=[sys.executable,str(HERE/'summarize_results.py'),'--input',str(output),'--output',str(output/'summary')]
        result=subprocess.run(summary_command,cwd=ROOT,env=env)
        status['summary_returncode']=result.returncode
        status['complete']=success and result.returncode==0;persist()
        if not status['complete']:raise SystemExit('Some stages failed. Inspect experiments.json and stage logs; --resume retries unfinished stages.')
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':main()
