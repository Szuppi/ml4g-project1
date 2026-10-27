import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
import json,hashlib,zipfile,sys,subprocess,shutil,time,random,platform
from pathlib import Path

MODEL='tree'
work=Path('/kaggle/working')
wheel_manifest=next(Path('/kaggle/input').rglob('wheel-manifest.json'))
wheels=work/'dependency-wheels';wheels.mkdir(exist_ok=True)
for name,expected in json.loads(wheel_manifest.read_text())['sha256'].items():
 source=wheel_manifest.parent/name
 if not source.exists():source=wheel_manifest.parent/name.replace('+','')
 assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
 shutil.copy2(source,wheels/name)
subprocess.run([sys.executable,'-m','pip','install','--quiet','--no-index','--find-links',str(wheels),'torch==2.14.1+cpu','numpy==2.4.6','pandas==3.0.6','scipy==1.17.1','scikit-learn==1.9.1','catboost==1.2.10'],check=True)
shutil.rmtree(wheels)
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from catboost import CatBoostRegressor

manifest_path=next(Path('/kaggle/input').rglob('prediction-input-manifest.json'));inputs=manifest_path.parent
manifest=json.loads(manifest_path.read_text())
for name,expected in manifest.items():
 if name=='model-code.zip' and not (inputs/name).exists():continue
 assert hashlib.sha256((inputs/name).read_bytes()).hexdigest()==expected
code=work/'code';code.mkdir(exist_ok=True)
if (inputs/'model-code.zip').exists():
 with zipfile.ZipFile(inputs/'model-code.zip') as archive:archive.extractall(code)
else:code=next(inputs.rglob('neural.py')).parent
assert hashlib.sha256((code/'neural.py').read_bytes()).hexdigest()=='fd65e8a19123ea708d2e66372a8524b0ae290eb936ecc36da0717ebc8e26a4d1'
sys.path.insert(0,str(code))
from neural import _preprocess,_predict,_Regressor
m=pd.read_csv(inputs/'metadata.csv');train=m.cell_line.isin(['X1','X2']).to_numpy();test=m.cell_line.eq('X3').to_numpy()
assert m.loc[test,'gex'].isna().all() and m.loc[test,'chr'].eq('chr1').all()
assert not m.loc[train,'chr'].eq('chr1').any()
y=m.loc[train,'gex'].to_numpy();cells=m.loc[train,'cell_line'].to_numpy()
assert np.isfinite(y).all() and (y>=0).all()
target=np.empty(len(y),np.float32)
for cell in np.unique(cells):
 mask=cells==cell;ref=np.sort(y[mask]);target[mask]=(np.searchsorted(ref,y[mask],'left')+np.searchsorted(ref,y[mask],'right'))/(2*len(ref))

from scipy.stats import spearmanr,rankdata
from sklearn.preprocessing import StandardScaler
from torch import nn
import gc
upgrade_manifest=next(Path('/kaggle/input').rglob('upgrade-input-manifest.json'));upgrade_inputs=upgrade_manifest.parent
for name,expected in json.loads(upgrade_manifest.read_text()).items():assert hashlib.sha256((upgrade_inputs/name).read_bytes()).hexdigest()==expected
relative=np.load(upgrade_inputs/'relative.npy')
baseX=np.concatenate([np.load(inputs/'coarse.npy'),np.load(inputs/'bins.npy')],1).astype(np.float32)
upgradeX=np.concatenate([baseX,relative],1)
assert len(upgradeX)==len(m) and not np.isinf(upgradeX).any()
folds={'D1':['chr3','chr11'],'D2':['chr17','chr21','chr22']};excluded=['chr1','chr12','chr18'];alphas=[0.,.55,.65,.75,.85,.9,.95,1.]
output=work/'results';output.mkdir(exist_ok=True);campaign_start=time.monotonic()

def reset():
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(2)

def targets(mask):
 labels=m.loc[mask,'gex'].to_numpy();cells=m.loc[mask,'cell_line'].to_numpy();out=np.empty(len(labels),np.float32)
 assert np.isfinite(labels).all() and (labels>=0).all()
 for cell in np.unique(cells):
  ix=cells==cell;ref=np.sort(labels[ix]);out[ix]=(np.searchsorted(ref,labels[ix],'left')+np.searchsorted(ref,labels[ix],'right'))/(2*len(ref))
 return out

def metric(mask,p):
 cells=m.loc[mask,'cell_line'].to_numpy();labels=m.loc[mask,'gex'].to_numpy();scores={}
 assert np.isfinite(p).all()
 for cell in np.unique(cells):
  ix=cells==cell;scores[cell]=0. if np.ptp(p[ix])==0 else float(spearmanr(labels[ix],p[ix]).statistic)
 return scores,float(np.mean(list(scores.values())))

def calibration(tr,ev,p):
 zero=float(np.mean(m.loc[tr,'gex'].to_numpy()==0));grid=[]
 for alpha in alphas:
  calibrated=np.where(p<=alpha*zero,0.,p);s,macro=metric(ev,calibrated);grid.append(dict(alpha=alpha,worst=min(s.values()),macro=macro,scores=s))
 selected=max(grid,key=lambda r:r['worst']);return selected['alpha'],zero,grid

def save_eval(dest,tr,ev,p,alpha,zero,record):
 calibrated=np.where(p<=alpha*zero,0.,p);scores,macro=metric(ev,calibrated)
 f=m.loc[ev,['cell_line','gene_name','chr','gex']].copy();f['continuous_prediction']=p;f['prediction']=calibrated;f.to_csv(dest/'validation_predictions.csv',index=False)
 record.update(scores=scores,macro=macro,alpha=alpha,zero_fraction=zero,n_train=int(tr.sum()),n_validation=int(ev.sum()),seed=42,train_chromosomes=sorted(m.loc[tr,'chr'].unique().tolist()),validation_chromosomes=sorted(m.loc[ev,'chr'].unique().tolist()))
 assert not set(record['train_chromosomes']).intersection(record['validation_chromosomes']) and not set(record['train_chromosomes']).intersection(excluded)
 (dest/'metrics.json').write_text(json.dumps(record,indent=2));print(json.dumps(dict(event='validated',**record)),flush=True)
 return record

def production_save(dest,p,record):
 assert len(p)==1984 and np.isfinite(p).all() and np.ptp(p)>0
 f=m.loc[test,['gene_name']].copy();f['gex_predicted']=p;f.to_csv(dest/'predictions.csv',index=False)
 (dest/'execution.json').write_text(json.dumps(record,indent=2));print(json.dumps(dict(event='production_done',**record)),flush=True)

class DualPool(nn.Module):
 def forward(self,x):return torch.cat([nn.functional.adaptive_avg_pool1d(x,10),nn.functional.adaptive_max_pool1d(x,10)],1)

class CompactUpgrade(_Regressor):
 def __init__(self,tab_dim):
  super().__init__('rescnn',tab_dim,64,.3)
  self.profile[-2]=DualPool();self.head[0]=nn.Linear(64*20+64,128)

def cnn_inputs(tr,ev,upgraded):
 d=dict(np.load(inputs/'features.npz'));ti,ei,pre=_preprocess(d,tr,ev,'rescnn');del d;gc.collect()
 if upgraded:
  selected=np.r_[np.flatnonzero(tr),np.flatnonzero(ev)];raw=upgradeX[selected];missing=~np.isfinite(raw);logged=np.sign(np.where(missing,0.,raw))*np.log1p(np.abs(np.where(missing,0.,raw)))
  scaler=StandardScaler().fit(logged[:tr.sum()]);tab=np.concatenate([scaler.transform(logged),missing.astype(np.float32)],1).astype(np.float32)
  ti[0]=torch.from_numpy(tab[:tr.sum()]);ei[0]=torch.from_numpy(tab[tr.sum():]);pre.update(upgrade_tab_mean=scaler.mean_,upgrade_tab_scale=scaler.scale_)
 return ti,ei,pre

def cnn_fit(dest,tr,ev,upgraded,production=False,selection=None):
 reset();start=time.monotonic();ti,ei,pre=cnn_inputs(tr,ev,upgraded);model=CompactUpgrade(ti[0].shape[1]) if upgraded else _Regressor('rescnn',ti[0].shape[1],64,.3)
 loader=DataLoader(TensorDataset(*ti,torch.from_numpy(targets(tr))),batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(42));optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
 average=None;average_count=0;history=[]
 for epoch in range(1,9):
  model.train();total=0.;n=0;tick=time.monotonic()
  for batch in loader:
   optimizer.zero_grad(set_to_none=True);pred=model(*batch[:-1]);loss=nn.functional.mse_loss(pred,batch[-1]);assert torch.isfinite(loss);loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(norm);optimizer.step();total+=loss.item()*len(pred);n+=len(pred)
  if upgraded and epoch>=6:
   state=model.state_dict()
   if average is None:average={k:v.detach().clone() for k,v in state.items()}
   else:
    for k,v in state.items():average[k].add_(v.detach())
   average_count+=1
  row=dict(epoch=epoch,loss=total/n,epoch_seconds=time.monotonic()-tick,seconds=time.monotonic()-start);history.append(row);print(json.dumps(dict(event='epoch',kind='cnn',recipe='upgrade' if upgraded else 'control',fit=dest.name,**row)),flush=True)
 last_state={k:v.detach().clone() for k,v in model.state_dict().items()};states={'last':last_state}
 if upgraded:states['swa']={k:v/average_count for k,v in average.items()}
 records=[]
 for name,state in states.items():
  if production and name!=selection['checkpoint']:continue
  sub=dest/name;sub.mkdir(exist_ok=True);model.load_state_dict(state);p=_predict(model,ei,torch.device('cpu'),128).astype(np.float64)
  torch.save(dict(state_dict=state,upgraded=upgraded,tab_dim=ti[0].shape[1],checkpoint=name),sub/'model.pt');np.savez_compressed(sub/'preprocessing.npz',**pre)
  fresh=CompactUpgrade(ti[0].shape[1]) if upgraded else _Regressor('rescnn',ti[0].shape[1],64,.3);fresh.load_state_dict(torch.load(sub/'model.pt',weights_only=True)['state_dict']);np.testing.assert_allclose(_predict(fresh,ei,torch.device('cpu'),128),p,rtol=0,atol=1e-6)
  record=dict(kind='cnn',recipe='upgrade' if upgraded else 'control',checkpoint=name,parameters=sum(x.numel() for x in model.parameters()),seconds=time.monotonic()-start,epochs=8)
  if production:
   alpha=selection['alpha'];zero=float(np.mean(m.loc[tr,'gex'].to_numpy()==0));production_save(sub,np.where(p<=alpha*zero,0.,p),dict(**record,alpha=alpha,zero_fraction=zero,n_train=int(tr.sum()),n_predict=int(ev.sum()),selection=selection))
  else:
   alpha,zero,grid=calibration(tr,ev,p);record['calibration_grid']=grid;records.append(save_eval(sub,tr,ev,p,alpha,zero,record))
 pd.DataFrame(history).to_csv(dest/'epochs.csv',index=False)
 del model,optimizer,loader,ti,ei;gc.collect()
 return records

def tree_fit(dest,tr,ev,upgraded,production=False,selection=None):
 reset();start=time.monotonic();X=upgradeX if upgraded else baseX
 model=CatBoostRegressor(iterations=960,depth=5,learning_rate=.05,l2_leaf_reg=10,loss_function='RMSE',random_seed=42,thread_count=2,verbose=200,allow_writing_files=False,border_count=64,nan_mode='Min');model.fit(X[tr],targets(tr));p=model.predict(X[ev]);model.save_model(str(dest/'model.cbm'))
 fresh=CatBoostRegressor();fresh.load_model(str(dest/'model.cbm'));np.testing.assert_allclose(fresh.predict(X[ev]),p,rtol=0,atol=1e-12)
 record=dict(kind='tree',recipe='upgrade' if upgraded else 'control',seconds=time.monotonic()-start,iterations=960,features=X.shape[1])
 if production:
  alpha=selection['alpha'];zero=float(np.mean(m.loc[tr,'gex'].to_numpy()==0));production_save(dest,np.where(p<=alpha*zero,0.,p),dict(**record,alpha=alpha,zero_fraction=zero,n_train=int(tr.sum()),n_predict=int(ev.sum()),selection=selection));return []
 alpha,zero,grid=calibration(tr,ev,p);record['calibration_grid']=grid;return [save_eval(dest,tr,ev,p,alpha,zero,record)]

all_records=[]
for fold,chroms in folds.items():
 ev=m.cell_line.isin(['X1','X2']).to_numpy() & m.chr.isin(chroms).to_numpy();tr=m.cell_line.isin(['X1','X2']).to_numpy() & ~m.chr.isin(excluded+chroms).to_numpy()
 assert not (tr&ev).any() and not (ev&test).any()
 for upgraded in [False,True]:
  dest=output/'validation'/fold/('upgrade' if upgraded else 'control');dest.mkdir(parents=True,exist_ok=True)
  records=cnn_fit(dest,tr,ev,upgraded) if MODEL=='cnn' else tree_fit(dest,tr,ev,upgraded)
  for r in records:r.update(fold=fold)
  all_records+=records
summary=[]
for recipe in ['control','upgrade']:
 choices=['last','swa'] if MODEL=='cnn' and recipe=='upgrade' else ['last']
 for checkpoint in choices:
  selected=[r for r in all_records if r['recipe']==recipe and r.get('checkpoint','last')==checkpoint]
  assert len(selected)==2
  summary.append(dict(recipe=recipe,checkpoint=checkpoint,macro=float(np.mean([r['macro'] for r in selected])),scores={c:float(np.mean([r['scores'][c] for r in selected])) for c in ['X1','X2']},alpha=float(np.median([r['alpha'] for r in selected])),mean_seconds=float(np.mean([r['seconds'] for r in selected]))))
selected=max([s for s in summary if s['recipe']=='upgrade'],key=lambda s:s['macro'])
(output/'source-selection.json').write_text(json.dumps(dict(kind=MODEL,selected=selected,summary=summary),indent=2));print(json.dumps(dict(event='source_selected',kind=MODEL,selected=selected,summary=summary)),flush=True)
dest=output/'production';dest.mkdir(exist_ok=True)
if MODEL=='cnn':cnn_fit(dest,train,test,True,production=True,selection=selected)
else:tree_fit(dest,train,test,True,production=True,selection=selected)
record=dict(kind=MODEL,total_seconds=time.monotonic()-campaign_start,source_summary=summary,source_selected=selected,development_fits=4,production_fits=1)
(output/'campaign.json').write_text(json.dumps(record,indent=2));shutil.make_archive(str(work/(MODEL+'-upgrade-outputs')),'gztar',output);print(json.dumps(dict(event='campaign_complete',**record)),flush=True)
