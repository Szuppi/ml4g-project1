import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
import json,sys,time,hashlib,subprocess,shutil,random
W=Path('/kaggle/working'); start=time.monotonic()
wm=next(Path('/kaggle/input').rglob('wheel-manifest.json'));wh=W/'wheels';wh.mkdir(exist_ok=True)
for name,digest in json.loads(wm.read_text())['sha256'].items():
 p=wm.parent/name
 if not p.exists():p=wm.parent/name.replace('+','')
 assert hashlib.sha256(p.read_bytes()).hexdigest()==digest;shutil.copy2(p,wh/name)
subprocess.run([sys.executable,'-m','pip','install','--quiet','--no-index','--find-links',str(wh),'torch==2.14.1+cpu','numpy==2.4.6','pandas==3.0.6','scipy==1.17.1','scikit-learn==1.9.1'],check=True)
shutil.rmtree(wh)
import numpy as np,pandas as pd,torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from scipy.stats import spearmanr
source=next(Path('/kaggle/input').rglob('manifest.json')).parent
for name,digest in json.loads((source/'manifest.json').read_text()).items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest
sys.path.insert(0,str(source));from model import CNN,_preprocess,_predict
m=pd.read_csv(source/'metadata.csv');assert set(m.cell_line)=={'X1','X2'} and np.isfinite(m.gex).all()
with np.load(source/'features.npz') as z:data={k:z[k] for k in z.files}
torch.set_num_threads(2);torch.set_num_interop_threads(1)
class AdaptiveCNN(CNN):
 def __init__(self,tab_dim,variant):
  super().__init__(tab_dim);self.variant=variant
  if variant=='shape':self.profile[0]=nn.Conv1d(35,64,7,padding=3)
  if variant=='dilated':
   for i,d in [(2,1),(4,2),(5,4)]:self.profile[i].layers[3]=nn.Conv1d(64,64,5,padding=2*d,dilation=d)
 def forward(self,tab,profile):
  if self.variant=='gain' and self.training:
   p=profile[:,:7]*torch.exp(torch.randn(len(profile),7,1,device=profile.device)*.15);profile=torch.cat([p,profile[:,7:]],1)
  if self.variant=='shape':
   p=profile[:,:7];obs=1-profile[:,14:21];den=obs.sum(-1,keepdim=True).clamp_min(1);mean=(p*obs).sum(-1,keepdim=True)/den
   scale=(((p-mean).square()*obs).sum(-1,keepdim=True)/den).sqrt().clamp_min(.1)
   profile=torch.cat([profile,((p-mean)/scale).clamp(-6,6)*obs],1)
  return super().forward(tab,profile)
def targets(mask):
 q=m.loc[mask];out=np.empty(len(q),np.float32)
 for c in q.cell_line.unique():
  a=q.cell_line.eq(c).to_numpy();y=q.loc[a,'gex'].to_numpy();ref=np.sort(y);out[a]=(np.searchsorted(ref,y,'left')+np.searchsorted(ref,y,'right'))/(2*len(ref))
 return out
results=[];out=W/'results';out.mkdir(exist_ok=True)
for variant in ['control','gain','shape','dilated']:
 for split in ['development','production']:
  random.seed(42);np.random.seed(42);torch.manual_seed(42);tick=time.monotonic()
  val=m.chr.isin(['chr3','chr11']).to_numpy() if split=='development' else np.zeros(len(m),bool)
  train=(~m.chr.isin(['chr1','chr3','chr11','chr12','chr18'])).to_numpy() if split=='development' else np.ones(len(m),bool)
  evaluate=val if val.any() else np.arange(len(m))<1
  ti,vi,pre=_preprocess(data,train,evaluate,'rescnn');net=AdaptiveCNN(ti[0].shape[1],variant)
  cells=torch.from_numpy(m.loc[train,'cell_line'].map({'X1':0,'X2':1}).to_numpy(np.int64))
  loader=DataLoader(TensorDataset(*ti,torch.from_numpy(targets(train)),cells),batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(42))
  opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001);avg=None;count=0
  for epoch in range(1,9):
   net.train()
   for tab,profile,y,cell in loader:
    opt.zero_grad(set_to_none=True);p=net(tab,profile);loss=nn.functional.mse_loss(p,y);perm=torch.randperm(len(y));diff=y-y[perm];ok=(cell==cell[perm])&(diff!=0)
    if ok.any():loss=loss+.01*nn.functional.softplus(-torch.sign(diff[ok])*(p[ok]-p[perm][ok])/.1).mean()
    assert torch.isfinite(loss);loss.backward();assert torch.isfinite(nn.utils.clip_grad_norm_(net.parameters(),5));opt.step()
   if epoch>=6:
    if avg is None:avg={k:v.detach().clone() for k,v in net.state_dict().items()}
    else:
     for k,v in net.state_dict().items():avg[k].add_(v.detach())
    count+=1
   print(json.dumps(dict(variant=variant,split=split,epoch=epoch,seconds=time.monotonic()-tick)),flush=True)
  net.load_state_dict({k:v/count for k,v in avg.items()});dest=out/variant/split;dest.mkdir(parents=True,exist_ok=True)
  torch.save(dict(state_dict=net.state_dict(),tab_dim=ti[0].shape[1],variant=variant),dest/'model.pt');np.savez_compressed(dest/'preprocessing.npz',**pre)
  row=dict(variant=variant,split=split,train_rows=int(train.sum()),seconds=time.monotonic()-tick,epochs=8)
  if val.any():
   p=_predict(net,vi,torch.device('cpu'),128).astype(float);q=m.loc[val,['cell_line','gene_name','chr','gex']].copy();q['continuous_prediction']=p;q.to_csv(dest/'validation.csv',index=False)
   row['scores']={c:float(spearmanr(q.loc[q.cell_line.eq(c),'gex'],q.loc[q.cell_line.eq(c),'continuous_prediction']).statistic) for c in q.cell_line.unique()}
  results.append(row);(out/'summary.json').write_text(json.dumps(dict(fits=results,total_fits=8,seconds=time.monotonic()-start),indent=2));del net,ti,vi,loader
shutil.copy2(source/'model.py',out/'model.py');print('COMPLETE',flush=True)
