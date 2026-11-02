"""Generic total-variation comparison bounds and the shared-nuisance control of the shared-law audit.

The conditional joint/product TV bound uses ordinary bounded-payoff sensitivity
and the true-optimal performance-difference identity, without the capped-value
sharing construction. The root bound additionally uses expected posterior-TV
stability relative to a fixed independent reference law.
"""
from collections import defaultdict
from fractions import Fraction as F
from functools import lru_cache
from itertools import product
import json
from pathlib import Path

from p2_index_robustness import Joint, Spec, latent_model, shrink, tv
from staged_pandora_exact import clear_caches


def product_marginals(m, h):
    out = {(): F(1)}
    for i in range(m.n):
        out = {paths+(path,): w*p for paths,w in out.items() for path,p in m.marginal(h,i).items()}
    return out


def joint_remaining(m, h):
    out = defaultdict(F)
    for paths,w in m.matching(h):
        out[tuple(path[len(prefix):] for path,prefix in zip(paths,h))] += w
    return dict(out)


def generic_comparison(m):
    @lru_cache(None)
    def dependence(h):
        return tv(joint_remaining(m,h),product_marginals(m,h))

    @lru_cache(None)
    def local_envelope(h):
        remaining = sum(len(s.costs)-len(prefix) for s,prefix in zip(m.specs,h))
        b = 1+remaining*m.cmax
        action = m.local_action(h)
        # A stop has the same value under both models; advances use two value
        # comparisons. This also counts the nonzero premature-stopping loss.
        bound = (b if action < 0 else 2*b)*dependence(h)
        gap = m.value(h)-m.actions(h)[action]
        assert 0 <= gap <= bound
        return bound

    @lru_cache(None)
    def occupancy(h):
        i = m.local_action(h)
        return local_envelope(h)+(F(0) if i < 0 else sum(
            p*occupancy(nxt) for p,nxt,_ in m.children(h,i)))

    for h in m.all_histories():
        local_envelope(h)
    delta = dependence(m.root)
    b = 1+m.horizon*m.cmax
    root_bound = 4*m.horizon*(m.n+1)*b*delta
    regret = m.value(m.root)-m.local_value(m.root)
    assert regret <= occupancy(m.root) <= root_bound
    return dict(root_joint_product_tv=delta,regret=regret,
                generic_occupancy_bound=occupancy(m.root),generic_root_tv_bound=root_bound)


def nuisance_model():
    # Shared nuisance signal U; independent Bernoulli(4/5) terminal rewards.
    worlds=[]
    for u,a,b in product((F(0),F(1)),repeat=3):
        w=F(1,2)*(F(4,5) if a else F(1,5))*(F(4,5) if b else F(1,5))
        worlds.append((((u,a),(u,b)),w))
    spec=Spec((F(1,100),F(1,5)),(False,True))
    return Joint(worlds,(spec,spec))


def main():
    out=Path('reports/p2_shared_law_controls')
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    p=F(4,5)
    for reliability in (F(1,2),F(3,4),F(9,10),F(1)):
        for lam in (F(0),F(1,100),F(1,10),F(1,2),F(1)):
            m=latent_model(shrink((F(4,5),F(1,5)),p,lam),
                           shrink((F(7,10),F(1,5)),p,lam),
                           reliability=(reliability,reliability))
            r=generic_comparison(m)
            assert r['root_joint_product_tv'] == F(12,125)*lam**2
            records.append(dict(reliability=reliability,lam=lam)|r)
            clear_caches()
    nuisance=nuisance_model()
    nr=nuisance.verify()|generic_comparison(nuisance)
    assert nr['root_joint_product_tv'] == F(1,2)
    assert nr['regret'] == nr['call_uniform'] == 0
    assert nr['tv_uniform'] == F(28,5)
    result=dict(cases=len(records),records=records,nuisance=nr,
                interpretation='Generic root-TV perturbation already gives O(lambda^2), including noiseless signals. Call envelope is stricter than full TV but is a value-aware/stop-loss specialization.')
    (out/'generic_comparison.json').write_text(json.dumps(result,default=str,indent=2)+'\n')
    print(json.dumps(dict(cases=len(records),nuisance=nr),default=str,indent=2))


if __name__ == '__main__':
    main()
