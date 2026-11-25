"""Normalised confusion matrices with Iqro' letter order and Arabic letters (from test_predictions.csv)."""
import sys, pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':['DejaVu Serif','DejaVu Sans'],'font.size':8})
ROOT=sys.argv[1]; OUT=sys.argv[2]
# Iqro' Volume 1 order = numeric prefix of the file names (01-A ... 29-YA)
ORDER=[('01a','Alif','ا'),('ba','ba','ب'),('ta','ta','ت'),('tsa','tsa','ث'),('ja','ja','ج'),('kha','kha','ح'),
       ('kho','kho','خ'),('da','da','د'),('dza','dza','ذ'),('ro','ro','ر'),('za','za','ز'),('sa','sa','س'),
       ('sya','sya','ش'),('sho','sho','ص'),('dho','dho','ض'),('tho','tho','ط'),('zho','zho','ظ'),("'a","'a",'ع'),
       ('gho','gho','غ'),('fa','fa','ف'),('qo','qo','ق'),('ka','ka','ك'),('la','la','ل'),('ma','ma','م'),
       ('na','na','ن'),('wa','wa','و'),('ha','ha','ه'),('28a','Hamzah','ء'),('ya','ya','ي')]
LAB=[o[0] for o in ORDER]; TXT=[f'{o[1]} {o[2]}' for o in ORDER]
def cm(path):
    d=pd.read_csv(path,keep_default_na=False)
    M=np.zeros((29,30))
    for r,p in zip(d.reference,d.prediction):
        M[LAB.index(r), LAB.index(p) if p in LAB else 29]+=1
    return M/M.sum(1,keepdims=True), int(d.correct.sum()), len(d)
def panel(pairs,titles,fn):
    fig,axs=plt.subplots(1,2,figsize=(10.2,5.3),constrained_layout=True)
    for ax,(p,t) in zip(axs,zip(pairs,titles)):
        N,c,n=cm(p); im=ax.imshow(N,cmap='Blues',vmin=0,vmax=1,aspect='auto')
        ax.set_title(f'{t} ({c}/{n} correct)',fontsize=9)
        ax.set_xticks(range(30)); ax.set_xticklabels(TXT+['Invalid'],rotation=90,fontsize=6.3)
        ax.set_yticks(range(29)); ax.set_yticklabels(TXT,fontsize=6.3)
        for tl in list(ax.get_xticklabels())+list(ax.get_yticklabels()):
            if tl.get_text().startswith(('Alif','Hamzah')): tl.set_fontweight('bold')
        ax.set_xlabel('Predicted class'); ax.set_ylabel('Reference class')
    fig.colorbar(im,ax=axs,shrink=0.85,label='Proportion of reference class')
    fig.savefig(fn,dpi=300)
panel([f'{ROOT}/whisper/Skenario 1/test_predictions.csv',f'{ROOT}/whisper/Skenario 3/test_predictions.csv'],
      ['(a) Whisper-tiny, Base','(b) Whisper-tiny, +log-Mel'],f'{OUT}/fig_cm_whisper.png')
panel([f'{ROOT}/wavlm/Skenario 1/test_predictions.csv',f'{ROOT}/wavlm/Skenario 2/test_predictions.csv'],
      ['(a) WavLM, Base','(b) WavLM, +MFCC'],f'{OUT}/fig_cm_wavlm.png')
