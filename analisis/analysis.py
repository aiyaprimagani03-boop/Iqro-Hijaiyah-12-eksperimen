import pandas as pd, numpy as np, glob, re, json, itertools
from scipy.stats import binomtest, norm
from collections import Counter
import sys
ROOT=sys.argv[1] if len(sys.argv)>1 else 'hasil/Test Predictions'
BB={'wav2vec':'wav2vec 2.0','Hubert':'HuBERT','wavlm':'WavLM','whisper':'Whisper-tiny'}
SC={'Skenario 1':'Base','Skenario 2':'+MFCC','Skenario 3':'+log-Mel'}
DG=['kh','sy','sh','ts','th','dz','dh','zh','gh']
def ph(s):
    s=re.sub(r'\d','',str(s).lower());o=[];i=0
    while i<len(s):
        if s[i:i+2] in DG:o.append(s[i:i+2]);i+=2
        else:o.append(s[i]);i+=1
    return o
def ed(a,b):
    d=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        p=d[:];d[0]=i
        for j,y in enumerate(b,1):d[j]=min(p[j]+1,d[j-1]+1,p[j-1]+(x!=y))
    return d[-1]
def wilson(k,n,z=1.959963984540054):
    p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den;h=z/den*np.sqrt(p*(1-p)/n+z*z/(4*n*n));return c-h,c+h
D={}
for b in BB:
    for s in SC:
        d=pd.read_csv(f'{ROOT}/{b}/{s}/test_predictions.csv',keep_default_na=False)
        D[(BB[b],SC[s])]=d.sort_values('filename').reset_index(drop=True)
keys=list(D)
fn0=D[keys[0]].filename.tolist()
assert all(D[k].filename.tolist()==fn0 for k in keys)
CLASSES=sorted(D[keys[0]].reference.unique()); assert len(CLASSES)==29
res={}
rng=np.random.default_rng(42)
spk=D[keys[0]].speaker_id.values
idx_by_spk={s:np.where(spk==s)[0] for s in np.unique(spk)}
B=10000
boot_idx=[np.concatenate([rng.choice(ix,len(ix),replace=True) for ix in idx_by_spk.values()]) for _ in range(B)]
boot_idx=np.array(boot_idx)
for k in keys:
    d=D[k];n=len(d);c=int(d.correct.sum())
    # word-level WER
    we=sum(ed(str(r).split(),str(p).split()) for r,p in zip(d.reference,d.prediction));wn=sum(len(str(r).split()) for r in d.reference)
    ce=sum(ed(list(str(r)),list(str(p))) for r,p in zip(d.reference,d.prediction));cn=sum(len(str(r)) for r in d.reference)
    pe=sum(ed(ph(r),ph(p)) for r,p in zip(d.reference,d.prediction));pn=sum(len(ph(r)) for r in d.reference)
    multiword=int(sum(len(str(p).split())!=1 for p in d.prediction))
    lo,hi=wilson(c,n)
    corr=d.correct.values
    bacc=corr[boot_idx].mean(1)
    # macro P/R/F1 over 29 classes; invalid predictions are wrong and never count as a predicted class
    P=[];R=[];F=[]
    for cl in CLASSES:
        tp=int(((d.reference==cl)&(d.prediction==cl)).sum());fp=int(((d.reference!=cl)&(d.prediction==cl)).sum());fn_=int(((d.reference==cl)&(d.prediction!=cl)).sum())
        p_=tp/(tp+fp) if tp+fp else 0.0;r_=tp/(tp+fn_) if tp+fn_ else 0.0;f_=2*p_*r_/(p_+r_) if p_+r_ else 0.0
        P.append(p_);R.append(r_);F.append(f_)
    alif=d.reference.isin(['01a','28a']);swap=int(((d.reference=='01a')&(d.prediction=='28a')).sum()+((d.reference=='28a')&(d.prediction=='01a')).sum())
    merged_correct=c+swap
    per_spk={s:float(corr[ix].mean()*100) for s,ix in idx_by_spk.items()}
    res[k]=dict(n=n,correct=c,acc=100*c/n,wil=(100*lo,100*hi),boot=(np.percentile(bacc,2.5)*100,np.percentile(bacc,97.5)*100),
        WER=100*we/wn,CER=100*ce/cn,PER=100*pe/pn,invalid=100*(1-d.is_valid_class.mean()),multiword=multiword,
        macroP=100*np.mean(P),macroR=100*np.mean(R),macroF=100*np.mean(F),
        alif_correct=int(((d.reference=='01a')&(d.prediction=='01a')).sum()),ham_correct=int(((d.reference=='28a')&(d.prediction=='28a')).sum()),
        swaps=swap,ah_invalid=int((alif&(d.is_valid_class==0)).sum()),errors=n-c,ah_err=int((alif&(d.correct==0)).sum()),
        acc_other=100*d[~alif].correct.mean(),merged_acc=100*merged_correct/n,per_spk=per_spk)
    res[k]['ah_share']=100*res[k]['ah_err']/res[k]['errors'] if res[k]['errors'] else float('nan')
for k in keys:
    r=res[k];print(f"{k[0]:12s} {k[1]:8s} c={r['correct']:3d} acc={r['acc']:.2f} W[{r['wil'][0]:.1f},{r['wil'][1]:.1f}] Bs[{r['boot'][0]:.1f},{r['boot'][1]:.1f}] WER={r['WER']:.2f} CER={r['CER']:.2f} PER={r['PER']:.2f} inv={r['invalid']:.2f} mw={r['multiword']} P/R/F={r['macroP']:.1f}/{r['macroR']:.1f}/{r['macroF']:.1f} spk={ {s:round(v,2) for s,v in r['per_spk'].items()} } merged={r['merged_acc']:.2f}")
print()
for k in keys:
    r=res[k]
    if r['correct']==0: continue
    print(f"{k[0]:12s} {k[1]:8s} alif={r['alif_correct']} ham={r['ham_correct']} swaps={r['swaps']} ahinv={r['ah_invalid']} ah_err={r['ah_err']} share={r['ah_share']:.1f} other={r['acc_other']:.2f} gap={abs(r['per_spk']['S02']-r['per_spk']['S09']):.1f}")
# McNemar within backbone
pairs=[('Base','+MFCC'),('Base','+log-Mel'),('+MFCC','+log-Mel')]
rows=[]
for bb in BB.values():
    for a,b_ in pairs:
        x=D[(bb,a)].correct.values;y=D[(bb,b_)].correct.values
        b10=int(((x==1)&(y==0)).sum());b01=int(((x==0)&(y==1)).sum())
        p=binomtest(min(b10,b01),b10+b01,0.5).pvalue if b10+b01 else 1.0
        p=min(1.0,p)
        diff=(y.mean()-x.mean())*100
        bd=(y[boot_idx].mean(1)-x[boot_idx].mean(1))*100
        # conditional odds ratio (second better / first better) with exact CI from Clopper-Pearson on b01/(b10+b01)
        m=b10+b01
        if m:
            ci=binomtest(b01,m).proportion_ci(0.95,'exact');lo_,hi_=ci.low,ci.high
            OR=b01/b10 if b10 else float('inf');ORlo=lo_/(1-lo_) if lo_<1 else float('inf');ORhi=hi_/(1-hi_) if hi_<1 else float('inf')
            g=b01/m-0.5
        else: OR=ORlo=ORhi=float('nan');g=0
        rows.append(dict(bb=bb,cmp=f'{a} vs {b_}',b10=b10,b01=b01,p=p,diff=diff,bci=(np.percentile(bd,2.5),np.percentile(bd,97.5)),OR=OR,ORci=(ORlo,ORhi),g=g))
# Holm
ps=np.array([r['p'] for r in rows]);order=np.argsort(ps);m=len(ps);adj=np.empty(m);run=0
for rank,i in enumerate(order):
    run=max(run,min(1,(m-rank)*ps[i]));adj[i]=run
for r,a in zip(rows,adj): r['holm']=a
print()
for r in rows: print(f"{r['bb']:12s} {r['cmp']:18s} b10={r['b10']:3d} b01={r['b01']:3d} p={r['p']:.4g} holm={r['holm']:.4g} diff={r['diff']:+.2f} boot=[{r['bci'][0]:+.1f},{r['bci'][1]:+.1f}] OR={r['OR']:.2f} [{r['ORci'][0]:.2f},{r['ORci'][1]:.2f}] g={r['g']:+.3f}")
# extra comparisons in text
for a,b_ in [(('Whisper-tiny','Base'),('wav2vec 2.0','+MFCC')),(('Whisper-tiny','Base'),('HuBERT','Base'))]:
    x=D[a].correct.values;y=D[b_].correct.values;b10=int(((x==1)&(y==0)).sum());b01=int(((x==0)&(y==1)).sum())
    print('extra',a,b_,b10,b01,binomtest(min(b10,b01),b10+b01).pvalue)
# confusions summed over 11 working experiments
conf=Counter();tot=Counter();percls=Counter()
work=[k for k in keys if res[k]['correct']>0]
for k in work:
    d=D[k]
    for r_,p_ in zip(d.reference,d.prediction):
        tot[r_]+=1
        if r_==p_: percls[r_]+=1
        else: conf[(r_,p_ if p_ in CLASSES else '<invalid>')]+=1
print('\nTop confusions:',[(f'{a}->{b}',v) for (a,b),v in conf.most_common(14)])
print('lowest class acc:',sorted([(c,round(100*percls[c]/tot[c],1)) for c in CLASSES],key=lambda x:x[1])[:6])
# Whisper logmel  'a / alif / hamzah predictions
d=D[('Whisper-tiny','+log-Mel')];print('Wh logmel:',{c:Counter(d[d.reference==c].prediction) for c in ["'a",'01a','28a']})
print('WavLM base invalid',int((D[('WavLM','Base')].is_valid_class==0).sum()),'mfcc',int((D[('WavLM','+MFCC')].is_valid_class==0).sum()))
print('class counts test',Counter(D[keys[0]].reference).most_common()[-3:])
json.dump({f'{k[0]}|{k[1]}':{kk:(vv if not isinstance(vv,tuple) else list(vv)) for kk,vv in v.items()} for k,v in res.items()},open('res.json','w'),indent=1,default=float)
json.dump(rows,open('mcnemar.json','w'),indent=1,default=float)
