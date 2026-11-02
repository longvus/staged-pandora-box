#!/usr/bin/env python3
"""Certify the collection-induced acquisition-switching example.

Verifies piecewise-affine Bellman certificates for every prior in a fixed
finite model and at sampled costs in the cost band; p2_switching_certificate.py
certifies the whole cost interval.
"""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

from p2_switching_scout import Model, Parametric, Scalar, at, line_at


def model(cost=F(1, 5), collect=(True, True)):
    return Model(a=(F(4, 5), F(1, 5)), b=(F(7, 10), F(1, 5)),
                 reliability=(F(3, 4),)*2, signal_cost=(F(1, 100),)*2,
                 terminal_cost=(cost,)*2, collect=collect)


def certificate(parametric):
    """Check affine Bellman identities/inequalities on all partition cells.

    At union breakpoints every expression is affine on each interval.
    Endpoint inequalities imply the inequalities everywhere on that interval;
    one action attaining both endpoints proves the Bellman maximum identity.
    Finite-horizon induction then identifies this certificate with the true
    value function for the given model, for every prior p in [0,1].
    """
    parametric.value(((), ()))
    result = []
    interval_checks = 0
    for history, actions in parametric.actions.items():
        value = parametric.value(history)
        functions = list(actions.values())+[value]
        for i, prefix in enumerate(history):
            if len(prefix) < 2:
                for y in (0, 1):
                    nxt = history[:i]+(prefix+(y,),)+history[i+1:]
                    functions.append(parametric.value(nxt))
        bounds = sorted({x for fn in functions for seg in fn for x in seg[:2]})
        l0, l1 = parametric.likelihood(history)
        for lo, hi in zip(bounds, bounds[1:]):
            for p in (lo, hi):
                mass = (1-p)*l0+p*l1
                assert at(actions["stop"], p) == mass*parametric.model.incumbent(history)
                for i, prefix in enumerate(history):
                    if len(prefix) == 2:
                        continue
                    expected = -mass*parametric.model.cost(i, prefix)
                    for y in (0, 1):
                        nxt = history[:i]+(prefix+(y,),)+history[i+1:]
                        expected += at(parametric.value(nxt), p)
                    assert at(actions[str(i)], p) == expected
                assert all(at(value, p) >= at(fn, p) for fn in actions.values())
            assert any(at(value, lo) == at(fn, lo) and at(value, hi) == at(fn, hi) for fn in actions.values())
            interval_checks += 1
        result.append(dict(history=history, likelihood=[str(l0), str(l1)],
                           incumbent=str(parametric.model.incumbent(history)),
                           value=[[str(x) for x in seg] for seg in value],
                           actions={key:[[str(x) for x in seg] for seg in fn] for key, fn in actions.items()}))
    return dict(states=len(result), affine_interval_checks=interval_checks, state_certificates=result)


def difference(parametric):
    actions=parametric.actions[((), ())]
    bounds=sorted({x for key in ("0", "1") for seg in actions[key] for x in seg[:2]})
    result=[]
    for lo, hi in zip(bounds, bounds[1:]):
        a,b=line_at(actions["0"],(lo+hi)/2)
        c,d=line_at(actions["1"],(lo+hi)/2)
        if result and result[-1][2:] == (a-c,b-d):
            result[-1]=(result[-1][0],hi,a-c,b-d)
        else:
            result.append((lo,hi,a-c,b-d))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    names={"terminal":(False,False),"mixed":(True,False),"all":(True,True)}
    certificates={}
    roots={}
    for name,collect in names.items():
        pa=Parametric(model(collect=collect))
        rs=pa.regions()
        cert=certificate(pa)
        points=sorted({x for actions in pa.actions.values() for fn in actions.values() for seg in fn for x in seg[:2]})
        probes=set(points)
        probes.update((lo+hi)/2 for lo,hi in zip(points,points[1:]))
        for p in probes:
            sc=Scalar(pa.model,p)
            assert sc.actions(((),())) == {key:at(fn,p) for key,fn in pa.actions[((),())].items()}
        cert["independent_scalar_checks"]=len(probes)
        cert["model"]=pa.model.json()
        cert["regions"]=[dict(lo=str(lo),hi=str(hi),actions=actions) for lo,hi,actions in rs]
        certificates[name]=cert
        roots[name]=pa
    expected=((F(0),F(15,19),("0",)),(F(15,19),F(16,19),("1",)),
              (F(16,19),F(12,13),("0",)),(F(12,13),F(1),("1",)))
    assert tuple(roots["all"].regions()) == expected
    assert roots["terminal"].regions() == [(F(0),F(1),("0",))]
    dif=difference(roots["all"])
    (args.out/"fixed_model_certificate.json").write_text(json.dumps(certificates,indent=2)+"\n")
    # Sampled-cost check. Each sampled cost has an all-prior exact certificate;
    # the whole cost interval is certified by p2_switching_certificate.py.
    cost_records=[]
    for j in range(21):
        c=F(19,100)+F(j,2000)
        pa=Parametric(model(cost=c))
        rs=pa.regions()
        r1=F(15,19)
        r2=(132-420*c)/(50+35*c)
        r3=(288-780*c)/(250-535*c)
        assert F(0)<r1<r2<r3<F(1)
        assert rs == [(F(0),r1,("0",)),(r1,r2,("1",)),(r2,r3,("0",)),(r3,F(1),("1",))]
        cert=certificate(pa)
        cost_records.append(dict(cost=str(c),roots=list(map(str,(r1,r2,r3))),
                                 certified_states=cert["states"],affine_interval_checks=cert["affine_interval_checks"]))
    (args.out/"sampled_cost_band.json").write_text(json.dumps(cost_records,indent=2)+"\n")
    report=dict(fixed_model_all_prior_bellman_certificate_passed=True,
                states_per_collection={key:value["states"] for key,value in certificates.items()},
                exact_all_collection_switches=["15/19","16/19","12/13"],
                difference_pieces=[[str(x) for x in seg] for seg in dif],
                sampled_cost_band=["19/100","1/5"],sampled_costs=21)
    (args.out/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    colors={"0":"#2878b5","1":"#e1812c","stop":"#999999"}
    fig,axes=plt.subplots(2,1,figsize=(8.2,5.8),gridspec_kw={"height_ratios":[1,2]},layout="constrained")
    for i,(name,pa) in enumerate(roots.items()):
        for lo,hi,actions in pa.regions():
            axes[0].broken_barh([(float(lo),float(hi-lo))],(i-.3,.6),facecolors=colors[actions[0]])
    axes[0].set_yticks((0,1,2),("Terminal only","A intermediate collectable","Both intermediate collectable"))
    axes[0].set_xlim(0,1)
    axes[0].set_xlabel("Prior p = P(Theta=1)")
    axes[0].legend(handles=[Patch(color=colors[k],label="Advance "+("A" if k=="0" else "B")) for k in ("0","1")],loc="upper left",bbox_to_anchor=(0,1.5),ncol=2,frameon=False)
    for lo,hi,a,b in dif:
        axes[1].plot([float(lo),float(hi)],[float(a+b*lo),float(a+b*hi)],color="#333333")
    for p,label,offset in ((F(15,19),"15/19",(-10,-24)),(F(16,19),"16/19",(0,14)),(F(12,13),"12/13",(10,-24))):
        axes[1].axvline(float(p),color="#999999",ls="--",lw=.8)
        axes[1].annotate(label,(float(p),0),xytext=offset,textcoords="offset points",ha="center",fontsize=8)
    axes[1].axhline(0,color="black",lw=.7)
    axes[1].set_xlabel("Prior p = P(Theta=1)")
    axes[1].set_ylabel("Optimal first-action value\nH_A(p) - H_B(p)")
    axes[1].set_xlim(0,1)
    axes[1].grid(alpha=.2)
    axes[1].set_title("Both intermediate stages collectable; c=.2, k=.01")
    fig.savefig(args.out/"collection_acquisition_witness.pdf",bbox_inches="tight",metadata={"CreationDate":None})
    fig.savefig(args.out/"collection_acquisition_witness.png",dpi=180,bbox_inches="tight")
    plt.close(fig)
    sources=args.out/"source_snapshot"
    sources.mkdir()
    for path in (Path(__file__),Path(__file__).with_name("p2_switching_scout.py"),Path(__file__).with_name("staged_pandora_exact.py")):
        (sources/path.name).write_bytes(path.read_bytes())
    manifest={str(p.relative_to(args.out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.out.rglob('*')) if p.is_file()}
    (args.out/"artifact_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
