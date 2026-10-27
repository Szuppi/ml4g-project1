import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
import json,hashlib,zipfile,sys,subprocess,shutil,time,random,platform
from pathlib import Path

MODEL='cnn'
work=Path(r'/home/szupernikusz/Projects/ZMSc_Semester_I/ML4Genomics/ml4g-project1/combined/work/ensemble/cnn')
wheel_manifest=next(Path(r'/home/szupernikusz/Projects/ZMSc_Semester_I/ML4Genomics/ml4g-project1/combined/work/staged').rglob('wheel-manifest.json'))
wheels=work/'dependency-wheels';wheels.mkdir(exist_ok=True)
for name,expected in json.loads(wheel_manifest.read_text())['sha256'].items():
 source=wheel_manifest.parent/name
 if not source.exists():source=wheel_manifest.parent/name.replace('+','')
 assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
 shutil.copy2(source,wheels/name)
None  # [localized] pip install skipped (conda env)
shutil.rmtree(wheels)
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from catboost import CatBoostRegressor

manifest_path=next(Path(r'/home/szupernikusz/Projects/ZMSc_Semester_I/ML4Genomics/ml4g-project1/combined/work/staged').rglob('prediction-input-manifest.json'));inputs=manifest_path.parent
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
upgrade_manifest=next(Path(r'/home/szupernikusz/Projects/ZMSc_Semester_I/ML4Genomics/ml4g-project1/combined/work/staged').rglob('upgrade-input-manifest.json'));upgrade_inputs=upgrade_manifest.parent
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


RECIPE='cnn_expressedrank'
hold=m.cell_line.isin(['X1','X2']).to_numpy() & m.chr.isin(['chr12','chr18']).to_numpy()
X=np.concatenate([baseX,relative[:,:70]],1) if RECIPE=='tree_smooth' else None

def fit(dest,tr,ev,production=False,selected=None):
 reset();start=time.monotonic();dest.mkdir(parents=True,exist_ok=True);states={};history=[]
 if RECIPE=='tree_smooth':
  model=CatBoostRegressor(iterations=960,depth=5,learning_rate=.05,l2_leaf_reg=30,loss_function='RMSE',random_seed=42,thread_count=2,verbose=200,allow_writing_files=False,border_count=64,nan_mode='Min',bootstrap_type='Bernoulli',subsample=.8)
  model.fit(X[tr],targets(tr));model.save_model(str(dest/'model.cbm'));states={'last':None}
  fresh=CatBoostRegressor();fresh.load_model(str(dest/'model.cbm'));np.testing.assert_allclose(fresh.predict(X[ev]),model.predict(X[ev]),rtol=0,atol=1e-12)
 else:
  d=dict(np.load(inputs/'features.npz'));ti,ei,pre=_preprocess(d,tr,ev,'rescnn');hi=None
  if not production:_,hi,hpre=_preprocess(d,tr,hold,'rescnn');assert set(pre)==set(hpre)
  del d;gc.collect();model=_Regressor('rescnn',ti[0].shape[1],64,.3)
  cell_ids=torch.from_numpy(m.loc[tr,'cell_line'].map({'X1':0,'X2':1}).to_numpy(dtype=np.int64));loader=DataLoader(TensorDataset(*ti,torch.from_numpy(targets(tr)),cell_ids),batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(42));optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
  zero_rank={ci:float(np.mean(m.loc[tr & m.cell_line.eq('X'+str(ci+1)).to_numpy(),'gex'].to_numpy()==0))/2 for ci in [0,1]}
  average=None;count=0
  for epoch in range(1,9):
   if RECIPE=='cnn_cosine':
    rate=.0001+.0009*(1+np.cos(np.pi*(epoch-1)/7))/2
    for group in optimizer.param_groups:group['lr']=rate
   model.train();total=0.;n=0;tick=time.monotonic()
   for batch in loader:
    optimizer.zero_grad(set_to_none=True);tab,profile=batch[:-2];label=batch[-2];cell=batch[-1]
    if RECIPE=='cnn_profilemask':
     # Mask one spatial assay in 15% of examples; retain coarse context.
     # Standardized signal/coverage become neutral; missing flags become true.
     profile=profile.clone();rows=torch.nonzero(torch.rand(len(label))<.15,as_tuple=True)[0];assays=torch.randint(7,(len(rows),))
     profile[rows,assays,:]=0.;profile[rows,assays+7,:]=0.;profile[rows,assays+14,:]=1.;profile[rows,assays+21,:]=1.
    pred=model(tab,profile);loss=nn.functional.mse_loss(pred,label);perm=torch.randperm(len(label));difference=label-label[perm];valid=(cell==cell[perm]) & (difference!=0)
    if valid.any():loss=loss+.01*nn.functional.softplus(-torch.sign(difference[valid])*(pred[valid]-pred[perm][valid])/.1).mean()
    if RECIPE=='cnn_expressedrank':
     # Additional O(batch) ordering loss among expressed genes within each line.
     for ci in [0,1]:
      ids=torch.nonzero((cell==ci)&(label>zero_rank[ci]+1e-6),as_tuple=True)[0]
      if len(ids)>1:
       pairs=ids[torch.randperm(len(ids))];delta=label[ids]-label[pairs];keep=delta!=0
       if keep.any():loss=loss+.0025*nn.functional.softplus(-torch.sign(delta[keep])*(pred[ids[keep]]-pred[pairs[keep]])/.1).mean()
    assert torch.isfinite(loss);loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(norm);optimizer.step();total+=loss.item()*len(pred);n+=len(pred)
   if epoch>=6:
    if average is None:average={k:v.detach().clone() for k,v in model.state_dict().items()}
    else:
     for k,v in model.state_dict().items():average[k].add_(v.detach())
    count+=1
   row=dict(event='epoch',recipe=RECIPE,fit=str(dest),epoch=epoch,seconds=time.monotonic()-start,epoch_seconds=time.monotonic()-tick,loss=total/n);history.append(row);print(json.dumps(row),flush=True)
  states={'last':{k:v.detach().clone() for k,v in model.state_dict().items()},'swa':{k:v/count for k,v in average.items()}}
  pd.DataFrame(history).to_csv(dest/'epochs.csv',index=False)
 records=[]
 for name,state in states.items():
  if production and name!=selected['checkpoint']:continue
  sub=dest/name;sub.mkdir(exist_ok=True)
  if RECIPE!='tree_smooth':
   model.load_state_dict(state);torch.save(dict(state_dict=state,tab_dim=ti[0].shape[1]),sub/'model.pt');np.savez_compressed(sub/'preprocessing.npz',**pre)
   fresh=_Regressor('rescnn',ti[0].shape[1],64,.3);fresh.load_state_dict(torch.load(sub/'model.pt',weights_only=True)['state_dict']);p=_predict(model,ei,torch.device('cpu'),128).astype(np.float64);np.testing.assert_allclose(_predict(fresh,ei,torch.device('cpu'),128),p,rtol=0,atol=1e-6)
  else:p=model.predict(X[ev])
  record=dict(recipe=RECIPE,checkpoint=name,seconds=time.monotonic()-start,parameters=0 if RECIPE=='tree_smooth' else sum(x.numel() for x in model.parameters()))
  if production:
   alpha=selected['alpha'];zero=float(np.mean(m.loc[tr,'gex'].to_numpy()==0));production_save(sub,np.where(p<=alpha*zero,0.,p),dict(**record,alpha=alpha,zero_fraction=zero,n_train=int(tr.sum()),selection=selected))
   cont=m.loc[ev,['gene_name']].copy();cont['gex_predicted']=p;cont.to_csv(sub/'continuous.csv',index=False)
  else:
   alpha,zero,grid=calibration(tr,ev,p);record['calibration_grid']=grid;records.append(save_eval(sub,tr,ev,p,alpha,zero,record))
   hp=model.predict(X[hold]) if RECIPE=='tree_smooth' else _predict(model,hi,torch.device('cpu'),128).astype(np.float64)
   h=m.loc[hold,['cell_line','gene_name','chr']].copy();h['continuous_prediction']=hp;h['prediction']=np.where(hp<=alpha*zero,0.,hp);assert np.isfinite(hp).all();h.to_csv(sub/'holdout_predictions.csv',index=False)
 del model;gc.collect()
 return records

records=[]
for fold,chroms in folds.items():
 tr=m.cell_line.isin(['X1','X2']).to_numpy() & ~m.chr.isin(excluded+chroms).to_numpy();ev=m.cell_line.isin(['X1','X2']).to_numpy() & m.chr.isin(chroms).to_numpy()
 assert not set(m.loc[tr,'chr']).intersection(m.loc[ev|hold,'chr']);records+=fit(output/'validation'/fold,tr,ev)
summary=[]
for checkpoint in sorted(set(r['checkpoint'] for r in records)):
 rows=[r for r in records if r['checkpoint']==checkpoint];summary.append(dict(recipe=RECIPE,checkpoint=checkpoint,macro=float(np.mean([r['macro'] for r in rows])),alpha=float(np.median([r['alpha'] for r in rows])),source_seconds=float(np.mean([r['seconds'] for r in rows]))))
selected=max(summary,key=lambda r:r['macro']);(output/'selection.json').write_text(json.dumps(dict(recipe=RECIPE,selected=selected,summary=summary),indent=2));print(json.dumps(dict(event='source_selected',selected=selected)),flush=True)
None  # [localized] skip production; ensemble X3 from committed artifacts
(output/'campaign.json').write_text(json.dumps(dict(recipe=RECIPE,fits=3,seconds=time.monotonic()-campaign_start),indent=2));print(json.dumps(dict(event='complete',recipe=RECIPE)),flush=True)
