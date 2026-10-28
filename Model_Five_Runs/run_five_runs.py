"""Execute the supplied Kvasir ResNet-50 experiment with five training seeds."""
import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
from pathlib import Path
import pandas as pd

METRICS = ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1')
SEEDS = [42, 123, 2026, 3407, 5678]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prepare_manifest(source, root, destination, check_files=True):
    df = pd.read_csv(source)
    required = {'path','class_name','label','split'}
    if not required.issubset(df.columns):
        raise ValueError('Manifest requires path,class_name,label,split')
    if df[list(required)].isna().any().any():
        raise ValueError('Missing manifest values')
    df['split'] = df['split'].replace({'val':'validation'})
    relative=[]
    for row in df.itertuples(index=False):
        parts=str(row.path).replace('\\','/').split('/')
        if len(parts)<2 or parts[-2]!=row.class_name:
            raise ValueError('Expected class/filename paths')
        relative.append('/'.join(parts[-2:]))
    if len(set(relative))!=len(relative):
        raise ValueError('Duplicate images in manifest')
    expected={'dyed-lifted-polyps':(700,200,100),'dyed-resection-margins':(350,100,50),
              'esophagitis':(700,200,100),'normal-cecum':(350,100,50),
              'normal-pylorus':(350,100,50),'normal-z-line':(350,100,50),
              'polyps':(700,200,100),'ulcerative-colitis':(700,200,100)}
    if set(df.class_name)!=set(expected):raise ValueError('Unexpected Kvasir classes')
    for label,(name,counts) in enumerate(expected.items()):
        rows=df[df.class_name==name]
        if not (rows.label==label).all():raise ValueError('Incorrect numeric class label')
        if tuple((rows.split==s).sum() for s in ('train','validation','test'))!=counts:
            raise ValueError('Split differs from original 6000-image protocol')
    df['path']=[str((root/r).resolve()) for r in relative]
    if check_files:
        missing=[x for x in df.path if not Path(x).is_file()]
        if missing:raise FileNotFoundError(f'{len(missing)} images missing; first: {missing[0]}')
    df.to_csv(destination,index=False)
    return df

def summarize(out, seeds):
    rows=[]
    for seed in seeds:
        folder=out/f'seed_{seed}';d=json.loads((folder/'test_metrics.json').read_text())
        if d['seed']!=seed or d['test_images']!=600:raise ValueError('Wrong seed or test sample count')
        for metric in METRICS:
            if not math.isfinite(d[metric]) or not 0<=d[metric]<=1:raise ValueError('Invalid metric')
        rows.append({'seed':seed,'test_images':600,**{m:d[m] for m in METRICS}})
    per_seed=pd.DataFrame(rows);per_seed.to_csv(out/'per_seed_metrics.csv',index=False)
    summary=[]
    for metric in METRICS:
        values=[row[metric] for row in rows]
        mean,sd=statistics.mean(values),statistics.stdev(values)
        summary.append({'metric':metric,'n_runs':len(seeds),'mean':mean,'sample_sd':sd,
                        'mean_percent':100*mean,'sample_sd_percentage_points':100*sd})
    pd.DataFrame(summary).to_csv(out/'mean_sample_sd.csv',index=False)
    (out/'repeated_run_summary.json').write_text(json.dumps({
        'dataset':'Kvasir','model':'ResNet-50','seeds':seeds,'test_images_per_run':600,
        'protocol':'independent training seeds, one fixed image split',
        'metric_units':'fractions; percentage columns multiply by 100',
        'standard_deviation':'sample SD, ddof=1','per_seed_results':rows,'summary':summary},indent=2))
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--manifest',default=str(Path(__file__).with_name('kvasir_fixed_manifest.csv')))
    p.add_argument('--seeds',default=','.join(map(str,SEEDS)))
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--prepare-only',action='store_true',help='Prepare commands without training or checking image availability')
    args=p.parse_args();seeds=[int(s) for s in args.seeds.split(',')]
    if len(seeds)!=5 or len(set(seeds))!=5:p.error('Exactly five unique seeds are required')
    if args.epochs<1:p.error('Epochs must be positive')
    out=Path(args.out).resolve();root=Path(args.root).resolve()
    if out.exists() and any(out.iterdir()):p.error('Choose a new or empty output folder')
    out.mkdir(parents=True,exist_ok=True)
    manifest=out/'fixed_manifest.csv';prepare_manifest(args.manifest,root,manifest,not args.prepare_only)
    script=Path(__file__).with_name('single_run_kvasir.py').resolve()
    commands=[]
    for seed in seeds:
        commands.append([sys.executable,str(script),'--root',str(root),'--manifest',str(manifest),
                         '--out',str(out/f'seed_{seed}'),'--seed',str(seed),'--epochs',str(args.epochs)])
    (out/'run_plan.json').write_text(json.dumps({'seeds':seeds,'epochs':args.epochs,
        'commands':commands,'python':platform.python_version(),'manifest_sha256':digest(manifest),
        'source_manifest_sha256':digest(args.manifest),'single_run_script_sha256':digest(script),
        'runner_sha256':digest(__file__),'prepare_only':args.prepare_only},indent=2))
    if args.prepare_only:
        print('Prepared fixed manifest and five commands. No training or results generated.');return
    for seed,command in zip(seeds,commands):
        folder=out/f'seed_{seed}';folder.mkdir()
        print(f'Starting seed {seed}; see {folder / "run.log"}',flush=True)
        with (folder/'run.log').open('w') as log:
            subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
        if digest(manifest)!=json.loads((out/'run_plan.json').read_text())['manifest_sha256']:
            raise RuntimeError('Fixed manifest changed during training')
    summarize(out,seeds)
    print(f'All five runs completed. Results: {out}')

if __name__=='__main__':main()
