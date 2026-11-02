#!/usr/bin/env python3
"""Run the protocol in reports/p2_index_robustness_2026-10-26/PROTOCOL.md."""
import argparse
from fractions import Fraction as F
import hashlib
from itertools import product
import json
from pathlib import Path
import random

from p2_index_robustness import Joint, Spec, latent_model, shrink
from staged_pandora_exact import clear_caches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=Path('reports/p2_index_robustness_2026-10-26'))
    args = ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    rng = random.Random(20261026)
    records = []

    def record(m, meta):
        r = m.verify()
        records.append(meta | r)
        clear_caches()

    lams = tuple(map(F, ('0','1/100','1/50','1/20','1/10','1/5','2/5','3/5','1')))
    bases = []
    for base in range(12):
        a,b = tuple(rng.sample([F(k,10) for k in (1,2,3,5,7,8,9)],2)), \
              tuple(rng.sample([F(k,10) for k in (1,2,3,5,7,8,9)],2))
        p = rng.choice((F(1,5),F(1,2),F(4,5)))
        rs = tuple(rng.choice((F(3,5),F(3,4),F(9,10))) for _ in range(2))
        costs = tuple((rng.choice((F(1,100),F(1,25),F(2,25))),
                       rng.choice((F(1,20),F(1,5),F(2,5)))) for _ in range(2))
        bases.append(dict(base=base,a=a,b=b,p=p,reliability=rs,costs=costs))
        for collect in product((False,True),repeat=2):
            for lam in lams:
                m = latent_model(shrink(a,p,lam),shrink(b,p,lam),reliability=rs,
                                 costs=costs,collect=collect,p=p)
                aa = [rs[i]/(1-rs[i])*abs((a,b)[i][1]-(a,b)[i][0]) for i in range(2)]
                bs = [1+2*m.cmax]*2
                big_m = max(aa[i]*sum(bs[j]*aa[j] for j in range(2) if j != i) for i in range(2))
                theory = m.horizon*lam**2*big_m
                result = m.verify()
                assert result['tv_uniform'] <= theory
                records.append(dict(kind='latent',base=base,collect=collect,lam=lam,
                                    quadratic_uniform=theory) | result)
                clear_caches()
    for base in range(12):
        worlds = []
        weights = [rng.randint(1,9) for _ in range(64)]
        for bits,w in zip(product((F(0),F(1)),repeat=6),weights):
            worlds.append(((bits[:3],bits[3:]),F(w,sum(weights))))
        specs = tuple(Spec((F(1,20),F(1,10),F(1,5)),
                           (bool((base+i)%2),bool((base+i)%3),True),
                           (((F(1),),F(1,4)),((F(0),F(1)),F(1,25)))) for i in range(2))
        record(Joint(worlds,specs),dict(kind='arbitrary',base=base))
    for lam in lams:
        p=F(4,5)
        m = latent_model(shrink((F(4,5),F(1,5)),p,lam),shrink((F(7,10),F(1,5)),p,lam))
        record(m,dict(kind='constructed',lam=lam))
    def serialize(x):
        if isinstance(x,F):
            return str(x)
        raise TypeError(type(x).__name__)
    (args.out/'records.json').write_text(json.dumps(records,default=serialize,indent=2)+'\n')
    (args.out/'models.json').write_text(json.dumps(bases,default=serialize,indent=2)+'\n')
    grid = [r for r in records if r['kind']=='latent']
    hashes = {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
              (Path('next_paper/run_p2_index_robustness.py'),Path('next_paper/p2_index_robustness.py'),
               Path('next_paper/staged_pandora_exact.py'),args.out/'PROTOCOL.md')}
    summary = dict(instances=len(records),grid_instances=len(grid),
                   histories=sum(r['states'] for r in records),actions=sum(r['actions'] for r in records),
                   grid_positive_regrets=sum(r['regret']>0 for r in grid),
                   grid_max_regret=max(r['regret'] for r in grid),
                   negative_correction_instances=sum(r['min_gamma']<0 for r in records),
                   independent_controls=sum(r['lam']==0 for r in grid),
                   max_independent_regret=max(r['regret'] for r in grid if r['lam']==0),
                   grid_bound_informative_below_one=sum(r['call_uniform']<1 for r in grid),
                   hashes=hashes)
    (args.out/'summary.json').write_text(json.dumps(summary,default=serialize,indent=2)+'\n')
    print(json.dumps(summary,default=serialize,indent=2))


if __name__ == '__main__':
    main()
