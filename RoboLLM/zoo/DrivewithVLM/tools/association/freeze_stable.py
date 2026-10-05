"""Freeze A0-A4 policy using validation only; test never selects policy."""
import argparse
import hashlib
import json
from pathlib import Path


def choose(summary):
    association={x['variant']:x for x in summary['association'] if x['split']=='val'
                 and x['stage']=='accepted' and x['class_group']=='all' and x['age_group']=='all'}
    geometry={x['variant']:x for x in summary['geometry'] if x['split']=='val'
              and x['kind']=='fused_common' and x['group']=='all'}
    baseline=association['A1'];selected='A1';checks={}
    for variant in ['A2','A3','A4']:
        current=association[variant]
        association_ok=(current['precision_judgable'] is not None
                        and current['precision_judgable']>=baseline['precision_judgable']
                        and current['recall']>=baseline['recall'])
        # Identical cohort only; a lower output count cannot improve this score.
        a,b=geometry.get('A1'),geometry.get(variant)
        geometry_ok=bool(a and b and a['count']>0 and a['count']==b['count']
                         and b['size_mae_m']['mean']<=a['size_mae_m']['mean']+1e-12
                         and b['size_mae_m']['p95']<=a['size_mae_m']['p95']+1e-12)
        strict_gain=bool(association_ok and geometry_ok and
                         (current['recall']>baseline['recall'] or b['size_mae_m']['mean']<a['size_mae_m']['mean']-1e-12))
        checks[variant]={'association_noninferiority':association_ok,
                         'cor_size_noninferiority_common_cohort':geometry_ok,'strict_gain':strict_gain}
        if strict_gain:selected=variant
    return selected,checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trainval-dir',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(args.trainval_dir)
    summary=json.loads((root/'summary.json').read_text());selected,checks=choose(summary)
    source=Path(__file__).parent
    record={'schema':'stable-fusion-frozen-policy-v1','selected_variant':selected,'checks':checks,
            'selection_split':'val','test_used_for_selection':False,'trainval_dir':str(root.resolve()),
            'summary_sha256':hashlib.sha256((root/'summary.json').read_bytes()).hexdigest(),
            'algorithm_sha256':hashlib.sha256((source/'stable_fusion.py').read_bytes()).hexdigest(),
            'algorithm_dependencies_sha256':{x:hashlib.sha256((source/x).read_bytes()).hexdigest() for x in ['core.py','run.py']},
            'selection_rule':'Compared with A1: accepted judgable precision and recall noninferior; common-cohort cor size mean/p95 noninferior; at least one strict gain',
            'limitations':['judgable precision excludes unknowns','cor often inherits ego geometry; size criterion is benchmark consistency only']}
    path=Path(args.output)
    with path.open('x') as stream:json.dump(record,stream,indent=2)
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
