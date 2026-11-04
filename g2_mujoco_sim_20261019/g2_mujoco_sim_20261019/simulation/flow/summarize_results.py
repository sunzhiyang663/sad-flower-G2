"""Summarize FM reports into per-report JSON/CSV, keeping unchecked results unknown."""
import argparse
import csv
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def summarize(directory,position_tolerance=.02,angle_tolerance=.02):
    rows,errors=[],[]
    for path in sorted(Path(directory).rglob('evaluation.json')):
        try:
            report=json.loads(path.read_text(encoding='utf-8'))
            candidates=report['candidates']
            requested=report.get('requested_candidates',len(candidates))
            numerical=0
            for candidate in candidates:
                checks=[candidate.get('endpoint_position_error_m'),candidate.get('endpoint_angle_error_rad'),
                    candidate.get('max_position_residual_m'),candidate.get('max_angle_residual_rad'),
                    candidate.get('speed_limit_excess_m_s'),candidate.get('yaw_limit_excess_rad_s')]
                numerical+=int(all(v is not None and math.isfinite(v) for v in checks) and
                    all(v<=limit for v,limit in zip(checks,[1e-5,1e-5,position_tolerance,angle_tolerance,1e-6,1e-6])))
            checked=[c for c in candidates if c.get('collision_checked') is True]
            row=dict(report=str(path.resolve()),split=report['settings']['split'],seed=report['settings'].get('seed'),
                checkpoint_sha256=report.get('checkpoint_sha256'),requested_candidates=requested,
                generated_candidates=len(candidates),generation_failures=max(0,requested-len(candidates)),
                numerical_pass_count=numerical,numerical_pass_rate=numerical/requested if requested else None,
                collision_checked_candidates=len(checked),
                collision_free_rate=None,execution_pass_rate=None)
            timings=report.get('inference_timings',[])
            if timings:
                row['mean_inference_batch_seconds']=sum(t['batch_seconds'] for t in timings)/len(timings)
                row['amortized_seconds_per_candidate']=sum(t['batch_seconds'] for t in timings)/sum(t['candidates'] for t in timings)
            for key,stats in report.get('summary',{}).items():
                row[key+'_mean']=stats['mean'];row[key+'_max']=stats['max']
            rows.append(row)
        except (KeyError,ValueError,TypeError) as exc:
            errors.append(dict(file=str(path),reason=str(exc)))
    navigation=[]
    for filename in ('navigation_report.json','validation.json'):
        for path in sorted(Path(directory).rglob(filename)):
            try:
                report=json.loads(path.read_text(encoding='utf-8'))
                navigation.append(dict(file=str(path.resolve()),type=filename,passed=report.get('passed'),
                                       planner=report.get('planner'),hardware_commands_sent=report.get('hardware_commands_sent')))
            except (ValueError,TypeError) as exc:
                errors.append(dict(file=str(path),reason=str(exc)))
    ledger=Path(directory)/'experiments.json'
    stages=[]
    if ledger.exists():
        try:
            experiment=json.loads(ledger.read_text(encoding='utf-8'))
            for seed,record in experiment['seeds'].items():
                for stage,history in record.items():
                    if isinstance(history,list) and history:
                        stages.append(dict(seed=seed,stage=stage,status=history[-1]['status'],
                            attempts=len(history),elapsed_seconds=sum(v.get('elapsed_seconds',0.) for v in history),
                            log=history[-1].get('log')))
        except (KeyError,ValueError,TypeError) as exc:
            errors.append(dict(file=str(ledger),reason=str(exc)))
    return dict(evaluations=rows,experiment_stages=stages,navigation_and_full_task=navigation,invalid_reports=errors,
        thresholds=dict(position_residual_m=position_tolerance,angle_residual_rad=angle_tolerance),
        note='Per-report statistics; no pooling across seeds/models/splits. Null means unmeasured. Numerical pass is not collision or execution pass. Overlapping windows are correlated.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--position-tolerance',type=float,default=.02)
    parser.add_argument('--angle-tolerance',type=float,default=.02)
    args=parser.parse_args()
    if min(args.position_tolerance,args.angle_tolerance)<=0:parser.error('Tolerances must be positive')
    result=summarize(ROOT/args.input,args.position_tolerance,args.angle_tolerance)
    output=ROOT/args.output
    output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    rows=result['evaluations']
    fields=sorted(set().union(*(row.keys() for row in rows))) if rows else ['report']
    with (output/'evaluations.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(evaluations=len(rows),invalid_reports=len(result['invalid_reports'])),indent=2))


if __name__=='__main__':main()
