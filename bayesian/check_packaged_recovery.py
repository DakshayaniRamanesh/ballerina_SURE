"""Live integration smoke for packaged core; not independent performance evidence."""
import argparse
from pathlib import Path
import json
from sure.recovery_core import recover
from sure.recovery_estimators import load_frozen
from sure.recovery_http import PaymentTransport
from .independent_evaluation import verify_freeze, client_at, design, SEED

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',default='http://127.0.0.1:9091'); args=p.parse_args()
    verify_freeze(); estimator,kernel=load_frozen(); client=client_at(args.base)
    out=Path('bayesian/outputs/packaged_smoke'); out.mkdir(parents=True,exist_ok=False)
    cases=design(SEED-2,True); count=0; totals={p:0 for p in ('v2','v4')}
    for case in cases:
        for policy in totals:
            ident=client.create_case(case)
            transport=PaymentTransport(args.base,ident,case['keyed'],client.session)
            r=recover(transport,kernel,estimator,policy,keyed=case['keyed'],cap=case['cap'],deadline=case['deadline'],budget=case['budget'])
            t=client.settled_truth(ident,1+r['recovery_calls'])
            row=dict(case=case['case'],episode=ident,family=case['family'],keyed=case['keyed'],truth=t,**r)
            with (out/'runs.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
            if t['effects']>1 or (r['final_observation']=='confirmed' and t['effects']!=1): raise RuntimeError('Ledger integrity violation')
            if r['recovery_calls']>case['budget'] or r['steps']>4 or (not case['keyed'] and any(x['action'] in ('retry','failover') for x in r['trace'])): raise RuntimeError('Resource/safety violation')
            totals[policy]+=r['client_confirmed_on_time'];count+=1
        print(f"Packaged smoke case {case['case']+1}/{len(cases)}",flush=True)
    (out/'complete.json').write_text(json.dumps(dict(rows=count,success=totals),indent=2),encoding='utf-8')
    print('Packaged smoke passed:',count,'episodes;',totals,'(not a performance comparison)')
if __name__=='__main__': main()
