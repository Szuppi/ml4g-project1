DEPTH=6

import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
import json,sys,time,hashlib,subprocess,shutil
W=Path('/kaggle/working');tick=time.monotonic();wm=next(Path('/kaggle/input').rglob('wheel-manifest.json'));wh=W/'wheels';wh.mkdir(exist_ok=True)
for name,digest in json.loads(wm.read_text())['sha256'].items():
 p=wm.parent/name
 if not p.exists():p=wm.parent/name.replace('+','')
 assert hashlib.sha256(p.read_bytes()).hexdigest()==digest;shutil.copy2(p,wh/name)
subprocess.run([sys.executable,'-m','pip','install','--quiet','--no-index','--find-links',str(wh),'numpy==2.4.6','pandas==3.0.6','scipy==1.17.1','catboost==1.2.10'],check=True);shutil.rmtree(wh)
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from catboost import CatBoostRegressor
mp=next(Path('/kaggle/input').rglob('prediction-input-manifest.json'));root=mp.parent
for name,digest in json.loads(mp.read_text()).items():
 if name=='model-code.zip' and not (root/name).exists():continue
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
up=next(Path('/kaggle/input').rglob('upgrade-input-manifest.json'));u=up.parent
for name,digest in json.loads(up.read_text()).items():assert hashlib.sha256((u/name).read_bytes()).hexdigest()==digest
all_meta=pd.read_csv(root/'metadata.csv');known=all_meta.cell_line.isin(['X1','X2']).to_numpy();m=all_meta.loc[known].reset_index(drop=True);assert np.isfinite(m.gex).all()
X=np.concatenate([np.load(root/'coarse.npy')[known],np.load(root/'bins.npy')[known],np.load(u/'relative.npy')[known]],1).astype(np.float32);assert X.shape==(len(m),1565)
def targets(mask):
 q=m.loc[mask];out=np.empty(len(q),np.float32)
 for c in q.cell_line.unique():
  take=q.cell_line.eq(c).to_numpy();y=q.loc[take,'gex'].to_numpy();s=np.sort(y);out[take]=(np.searchsorted(s,y,'left')+np.searchsorted(s,y,'right'))/(2*len(s))
 return out
out=W/'results';out.mkdir(exist_ok=True);fits=[]
for protocol in ['x1_to_x2','x2_to_x1','joint','production']:
 start=time.monotonic();sources=['X1'] if protocol=='x1_to_x2' else ['X2'] if protocol=='x2_to_x1' else ['X1','X2'];train=(~m.chr.isin(['chr1','chr3','chr11','chr12','chr18'])&m.cell_line.isin(sources)).to_numpy() if protocol!='production' else np.ones(len(m),bool);val=m.chr.isin(['chr3','chr11']).to_numpy();dest=out/protocol;dest.mkdir(exist_ok=True)
 model=CatBoostRegressor(iterations=1000,depth=DEPTH,learning_rate=.05,l2_leaf_reg=20,loss_function='RMSE',random_seed=43,thread_count=2,verbose=200,allow_writing_files=False,border_count=128,nan_mode='Min',bootstrap_type='Bernoulli',subsample=.8)
 model.fit(X[train],targets(train));model.save_model(str(dest/'model.cbm'));row=dict(protocol=protocol,depth=DEPTH,train_rows=int(train.sum()),source_zero=float((m.loc[train,'gex']==0).mean()),seconds=time.monotonic()-start)
 if protocol!='production':
  p=model.predict(X[val]);assert np.isfinite(p).all();f=m.loc[val,['cell_line','gene_name','chr','gex']].copy();f['continuous_prediction']=p;f.to_csv(dest/'validation.csv',index=False);row['scores']={c:float(spearmanr(g.gex,g.continuous_prediction).statistic) for c,g in f.groupby('cell_line')}
 fits.append(row);(out/'summary.json').write_text(json.dumps(dict(fits=fits,total_fits=4,seconds=time.monotonic()-tick),indent=2));print(json.dumps(row),flush=True)
print('COMPLETE',flush=True)
