"""Train one leakage-controlled outer fold and emit item-level predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

from ed_stance_model import EDStanceConfig, EDStanceModel
from hard_pairs import mine
from splits_and_metrics import make_development_folds, stable_text_id, tune_threshold


SYSTEMS = {
    "independent": dict(dependency=False, contrastive=False, random_negative=False),
    "dependency": dict(dependency=True, contrastive=False, random_negative=False),
    "dependency_random": dict(dependency=True, contrastive=True, random_negative=True),
    "dependency_hard": dict(dependency=True, contrastive=True, random_negative=False),
}


class TweetDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128, labels=True):
        self.df=df.reset_index(drop=True);self.tokenizer=tokenizer;self.max_length=max_length;self.labels=labels
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r=self.df.iloc[i]; item=self.tokenizer(str(r.text_orig),truncation=True,max_length=self.max_length)
        item["doc_id"]=int(r.Column1)
        if self.labels:
            item["suffering_labels"]=float(r.label_suffering);item["promotion_labels"]=float(r.label_promoting)
        return item


class Collator:
    def __init__(self, tokenizer): self.base=DataCollatorWithPadding(tokenizer,return_tensors="pt")
    def __call__(self, batch):
        doc=torch.tensor([x.pop("doc_id") for x in batch],dtype=torch.long)
        ys=torch.tensor([x.pop("suffering_labels") for x in batch],dtype=torch.float) if "suffering_labels" in batch[0] else None
        yp=torch.tensor([x.pop("promotion_labels") for x in batch],dtype=torch.float) if "promotion_labels" in batch[0] else None
        out=self.base(batch);out["doc_id"]=doc
        if ys is not None:out["suffering_labels"]=ys;out["promotion_labels"]=yp
        return out


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def tree_hash(root):
    h=hashlib.sha256();root=Path(root)
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name!="tapt_commit.json":h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def make_model(encoder, spec, cfg):
    config=EDStanceConfig(encoder_name=encoder,dropout=cfg["dropout"],use_dependency_head=spec["dependency"],use_hierarchy_penalty=False)
    return EDStanceModel(config)


def optimizer_for(model,cfg):
    head=[];encoder=[]
    for name,p in model.named_parameters():
        (encoder if name.startswith("encoder.") else head).append(p)
    return torch.optim.AdamW([{"params":encoder,"lr":cfg["learning_rate"]},{"params":head,"lr":1e-4}],weight_decay=cfg["weight_decay"])


def to_device(batch,device): return {k:v.to(device) for k,v in batch.items() if k!="doc_id"}


def triplet_batch(pair_df, text_by_id, tokenizer, device, rng, size=8, max_length=128):
    if pair_df is None or pair_df.empty:return None
    idx=rng.choice(len(pair_df),size=min(size,len(pair_df)),replace=False);p=pair_df.iloc[idx]
    texts=[]
    for col in ["anchor_doc_id","positive_doc_id","negative_doc_id"]:texts.extend(text_by_id[int(x)] for x in p[col])
    return tokenizer(texts,padding=True,truncation=True,max_length=max_length,return_tensors="pt").to(device),len(p)


def fit(model,train_df,pairs,tokenizer,cfg,seed,device):
    generator=torch.Generator();generator.manual_seed(seed)
    ds=TweetDataset(train_df,tokenizer,cfg["max_length"],True)
    loader=DataLoader(ds,batch_size=cfg["batch_size"],shuffle=True,generator=generator,collate_fn=Collator(tokenizer))
    opt=optimizer_for(model,cfg);epochs=int(cfg["epochs"]);accum=int(cfg["gradient_accumulation_steps"])
    total_microbatches=len(loader)*epochs;total_updates=int(np.ceil(total_microbatches/accum));sched=get_linear_schedule_with_warmup(opt,int(total_updates*cfg["warmup_ratio"]),total_updates)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda");rng=np.random.default_rng(seed+991);model.train();opt.zero_grad(set_to_none=True)
    exposure=[];nonpadding=0;steps=0;triplet_steps=0;start=time.time()
    text_by_id=dict(zip(train_df.Column1.astype(int),train_df.text_orig.astype(str)))
    global_micro=0
    for epoch in range(epochs):
        for batch_i,batch in enumerate(loader):
            global_micro+=1
            exposure.extend(batch["doc_id"].tolist());nonpadding+=int(batch["attention_mask"].sum())
            labels={"suffering_labels":batch["suffering_labels"].to(device),"promotion_labels":batch["promotion_labels"].to(device)}
            inputs={k:v.to(device) for k,v in batch.items() if k in ("input_ids","attention_mask")}
            with torch.amp.autocast(device_type=device.type,enabled=device.type=="cuda"):
                out=model(**inputs,**labels);loss=out["loss"]
                if pairs is not None and global_micro%2==1:
                    trip=triplet_batch(pairs,text_by_id,tokenizer,device,rng,8,cfg["max_length"])
                    if trip:
                        encoded,ntrip=trip;z=model.encode_projection(**encoded);za,zp,zn=z[:ntrip],z[ntrip:2*ntrip],z[2*ntrip:]
                        con=F.softplus(((za*zn).sum(-1)-(za*zp).sum(-1))/cfg["contrastive_temperature"]).mean()
                        loss=loss+cfg["contrastive_loss_weight"]*con;triplet_steps+=1;nonpadding+=int(encoded.attention_mask.sum())
                remainder=total_microbatches%accum
                window_size=remainder if remainder and global_micro>total_microbatches-remainder else accum
                loss=loss/window_size
            scaler.scale(loss).backward()
            if global_micro%accum==0 or global_micro==total_microbatches:
                scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);scaler.step(opt);scaler.update();sched.step();opt.zero_grad(set_to_none=True);steps+=1
    counts=pd.Series(exposure).value_counts();assert len(counts)==len(train_df) and counts.min()==epochs and counts.max()==epochs
    assert steps==total_updates
    return {"epochs":epochs,"gold_rows":len(train_df),"gold_exposures":len(exposure),"min_exposures_per_id":int(counts.min()),
            "max_exposures_per_id":int(counts.max()),"optimizer_steps":steps,"triplet_steps":triplet_steps,
            "nonpadding_tokens":nonpadding,"wall_seconds":time.time()-start}


@torch.no_grad()
def predict(model,df,tokenizer,cfg,device):
    model.eval();ds=TweetDataset(df,tokenizer,cfg["max_length"],False);loader=DataLoader(ds,batch_size=cfg["batch_size"]*2,shuffle=False,collate_fn=Collator(tokenizer))
    ids=[];ls=[];lp=[]
    for b in loader:
        ids.extend(b.pop("doc_id").tolist());inputs={k:v.to(device) for k,v in b.items()};out=model(**inputs)
        ls.extend(out["suffering_logits"].float().cpu().tolist());lp.extend(out["promotion_logits"].float().cpu().tolist())
    return pd.DataFrame({"doc_id":ids,"logit_suffering":ls,"logit_promoting":lp}).assign(
        prob_suffering=lambda x:1/(1+np.exp(-x.logit_suffering)),prob_promoting=lambda x:1/(1+np.exp(-x.logit_promoting)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--core",required=True);ap.add_argument("--fold-manifest",required=True);ap.add_argument("--config",required=True)
    ap.add_argument("--outer-fold",type=int,required=True);ap.add_argument("--system",choices=SYSTEMS,required=True);ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--encoder",default=None);ap.add_argument("--output-dir",required=True);args=ap.parse_args()
    cfg=json.load(open(args.config));spec=SYSTEMS[args.system];encoder=args.encoder or cfg["encoder_name"];seed_all(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");tokenizer=AutoTokenizer.from_pretrained(encoder,normalization=True,use_fast=False)
    core=pd.read_csv(args.core);manifest=pd.read_csv(args.fold_manifest)
    if manifest.doc_id.duplicated().any() or core.Column1.duplicated().any():raise ValueError("Duplicate document IDs")
    if set(manifest.outer_fold)!=set(range(5)) or set(core.Column1)!=set(manifest.doc_id):raise ValueError("Core/fold manifest mismatch")
    core=core.merge(manifest[["doc_id","outer_fold","text_hash","cluster_id","cluster_size"]],left_on="Column1",right_on="doc_id",how="left",validate="one_to_one")
    if core.outer_fold.isna().any():raise ValueError("Unassigned core row")
    outer_train=core[core.outer_fold.ne(args.outer_fold)].copy();test=core[core.outer_fold.eq(args.outer_fold)].copy()
    if set(outer_train.Column1)&set(test.Column1) or len(outer_train)+len(test)!=len(core):raise ValueError("Outer split overlap/incompleteness")
    inner=make_development_folds(outer_train,5,20260815+args.outer_fold);inner_train=outer_train[inner!=0].copy();val=outer_train[inner==0].copy()
    inner_pairs=mine(inner_train,args.seed,spec["random_negative"]) if spec["contrastive"] else None
    model=make_model(encoder,spec,cfg).to(device);inner_ledger=fit(model,inner_train,inner_pairs,tokenizer,cfg,args.seed,device);val_pred=predict(model,val,tokenizer,cfg,device)
    val_pred=val[["Column1","label_suffering","label_promoting"]].rename(columns={"Column1":"doc_id"}).merge(val_pred,on="doc_id",validate="one_to_one")
    ts=tune_threshold(val_pred.label_suffering,val_pred.prob_suffering);tp=tune_threshold(val_pred.label_promoting,val_pred.prob_promoting)
    # Retrain from the same pretrained checkpoint on every outer-training row for an identical four exposures.
    del model
    if torch.cuda.is_available():torch.cuda.empty_cache()
    seed_all(args.seed);outer_pairs=mine(outer_train,args.seed,spec["random_negative"]) if spec["contrastive"] else None
    model=make_model(encoder,spec,cfg).to(device);outer_ledger=fit(model,outer_train,outer_pairs,tokenizer,cfg,args.seed,device);pred=predict(model,test,tokenizer,cfg,device)
    pred=test[["Column1","text_hash","cluster_id","cluster_size","stream_group","label_suffering","label_promoting"]].rename(columns={"Column1":"doc_id"}).merge(pred,on="doc_id",validate="one_to_one")
    pred["pred_suffering"]=pred.prob_suffering.ge(ts).astype(int);pred["pred_promoting"]=pred.prob_promoting.ge(tp).astype(int)
    pred["outer_fold"]=args.outer_fold;pred["model_seed"]=args.seed;pred["system"]=args.system;pred["threshold_suffering"]=ts;pred["threshold_promoting"]=tp
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True);pred.to_csv(out/"predictions.csv",index=False);val_pred.to_csv(out/"validation_predictions.csv",index=False)
    torch.save(model.state_dict(),out/"model_state.pt")
    encoder_commit=Path(encoder)/"tapt_commit.json" if Path(encoder).is_dir() else None
    run={"system":args.system,"outer_fold":args.outer_fold,"seed":args.seed,"encoder":encoder,"encoder_revision":getattr(model.encoder.config,"_commit_hash",None),
         "encoder_commit_sha256":sha(encoder_commit) if encoder_commit and encoder_commit.exists() else None,"device":str(device),"thresholds":{"suffering":ts,"promoting":tp},
         "inner_ledger":inner_ledger,"outer_ledger":outer_ledger,"core_sha256":sha(args.core),"fold_manifest_sha256":sha(args.fold_manifest),
         "config_sha256":sha(args.config),"model_code_sha256":sha(Path(__file__).with_name("ed_stance_model.py")),"trainer_code_sha256":sha(__file__),
         "hard_pairs_code_sha256":sha(Path(__file__).with_name("hard_pairs.py")),"splits_code_sha256":sha(Path(__file__).with_name("splits_and_metrics.py")),
         "encoder_tree_sha256":tree_hash(encoder) if Path(encoder).is_dir() else None,"test_rows":len(test)}
    (out/"run.json").write_text(json.dumps(run,indent=2),encoding="utf-8");print(json.dumps(run,indent=2))


if __name__=="__main__":main()
