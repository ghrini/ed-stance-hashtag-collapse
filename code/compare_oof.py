"""Paired document-level development comparison; seeds are already ensembled."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, f1_score, hamming_loss


def ece(y,p,bins=10):
    order=np.argsort(p);chunks=np.array_split(order,bins);return float(sum(len(i)/len(y)*abs(p[i].mean()-y[i].mean()) for i in chunks if len(i)))


def score(d, prefix):
    fs=f1_score(d.label_suffering,d[f"{prefix}_pred_suffering"],zero_division=0)
    fp=f1_score(d.label_promoting,d[f"{prefix}_pred_promoting"],zero_division=0)
    ps=d[f"{prefix}_prob_suffering"].to_numpy();pp=d[f"{prefix}_prob_promoting"].to_numpy();ys=d.label_suffering.to_numpy();yp=d.label_promoting.to_numpy()
    joint_true=np.c_[ys,yp];joint_pred=np.c_[d[f"{prefix}_pred_suffering"],d[f"{prefix}_pred_promoting"]]
    return {"macro_f1":(fs+fp)/2,"suffering_f1":fs,"promoting_f1":fp,
            "suffering_auprc":average_precision_score(ys,ps),"promoting_auprc":average_precision_score(yp,pp),
            "macro_auprc":.5*(average_precision_score(ys,ps)+average_precision_score(yp,pp)),
            "suffering_brier":brier_score_loss(ys,ps),"promoting_brier":brier_score_loss(yp,pp),
            "suffering_ece":ece(ys,ps),"promoting_ece":ece(yp,pp),"exact_match":float(np.all(joint_true==joint_pred,axis=1).mean()),
            "hamming_loss":float(hamming_loss(joint_true,joint_pred))}


def binary_f1(y, pred):
    tp=np.count_nonzero((y==1)&(pred==1));fp=np.count_nonzero((y==0)&(pred==1));fn=np.count_nonzero((y==1)&(pred==0))
    denominator=2*tp+fp+fn
    return 0.0 if denominator==0 else 2*tp/denominator


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--baseline",required=True);ap.add_argument("--new",required=True);ap.add_argument("--output",required=True)
    ap.add_argument("--bootstrap",type=int,default=10000);ap.add_argument("--permutations",type=int,default=100000);ap.add_argument("--seed",type=int,default=20260815);args=ap.parse_args()
    a=pd.read_csv(args.baseline).rename(columns={c:f"base_{c}" for c in ["prob_suffering","prob_promoting","pred_suffering","pred_promoting","system"]})
    b=pd.read_csv(args.new).rename(columns={c:f"new_{c}" for c in ["prob_suffering","prob_promoting","pred_suffering","pred_promoting","system"]})
    keys=["doc_id","stream_group","label_suffering","label_promoting","outer_fold"]
    d=a.merge(b,on=keys,validate="one_to_one");
    if len(d)!=1932:raise ValueError(f"Expected 1932 paired rows, got {len(d)}")
    base=score(d,"base");new=score(d,"new");observed=new["macro_f1"]-base["macro_f1"]
    d=d.reset_index(drop=True);rng=np.random.default_rng(args.seed);strata=[x.index.to_numpy() for _,x in d.groupby(["outer_fold","stream_group"],sort=True)]
    boot=np.empty(args.bootstrap);boot_s=np.empty(args.bootstrap);boot_p=np.empty(args.bootstrap);boot_sa=np.empty(args.bootstrap);boot_pa=np.empty(args.bootstrap)
    for i in range(args.bootstrap):
        idx=np.concatenate([rng.choice(s,len(s),replace=True) for s in strata]);x=d.iloc[idx];bs=score(x,"base");ns=score(x,"new")
        boot[i]=ns["macro_f1"]-bs["macro_f1"];boot_s[i]=ns["suffering_f1"]-bs["suffering_f1"];boot_p[i]=ns["promoting_f1"]-bs["promoting_f1"]
        boot_sa[i]=ns["suffering_auprc"]-bs["suffering_auprc"];boot_pa[i]=ns["promoting_auprc"]-bs["promoting_auprc"]
    ys=d.label_suffering.to_numpy(int);yp=d.label_promoting.to_numpy(int)
    bps=d.base_pred_suffering.to_numpy(int);bpp=d.base_pred_promoting.to_numpy(int)
    nps=d.new_pred_suffering.to_numpy(int);npp=d.new_pred_promoting.to_numpy(int)
    extreme=0;done=0
    while done<args.permutations:
        n=min(1000,args.permutations-done)
        for _ in range(n):
            swap=rng.random(len(d))<.5
            ps_base=np.where(swap,nps,bps);pp_base=np.where(swap,npp,bpp)
            ps_new=np.where(swap,bps,nps);pp_new=np.where(swap,bpp,npp)
            delta=.5*((binary_f1(ys,ps_new)-binary_f1(ys,ps_base))+(binary_f1(yp,pp_new)-binary_f1(yp,pp_base)))
            extreme+=abs(delta)>=abs(observed)-1e-15
        done+=n
    streams={}
    for group,x in d.groupby("stream_group"):
        bs=score(x,"base");ns=score(x,"new");streams[str(group)]={"n":len(x),"baseline":bs,"new":ns,"delta_macro_f1":ns["macro_f1"]-bs["macro_f1"]}
    result={"baseline":base,"new":new,"delta_macro_f1":observed,"delta_macro_f1_ci":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
            "delta_suffering_f1":new["suffering_f1"]-base["suffering_f1"],"delta_suffering_f1_ci":[float(np.quantile(boot_s,.025)),float(np.quantile(boot_s,.975))],
            "delta_promoting_f1":new["promoting_f1"]-base["promoting_f1"],"delta_promoting_f1_ci":[float(np.quantile(boot_p,.025)),float(np.quantile(boot_p,.975))],
            "delta_suffering_auprc_ci":[float(np.quantile(boot_sa,.025)),float(np.quantile(boot_sa,.975))],
            "delta_promoting_auprc_ci":[float(np.quantile(boot_pa,.025)),float(np.quantile(boot_pa,.975))],
            "suffering_noninferior_margin":-.02,"suffering_noninferior":bool(np.quantile(boot_s,.025)>-.02),
            "paired_randomization_p":(extreme+1)/(args.permutations+1),"bootstrap_replicates":args.bootstrap,"permutations":args.permutations,"streams":streams,
            "resampling_seed":args.seed,"development_only":True,"selection_warning":"Any choice among multiple systems using this corpus is exploratory; only the sealed challenge is confirmatory."}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps(result,indent=2))


if __name__=="__main__":main()
