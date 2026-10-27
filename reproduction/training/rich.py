import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
import json,hashlib,zipfile,sys,subprocess,shutil,time,random,platform
from pathlib import Path

MODEL='catboost_rich5'
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
seed=42;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
begun=time.monotonic();output=work/'predictions';output.mkdir(exist_ok=True)
if MODEL=='catboost_rich5':
 rich_manifest=next(Path('/kaggle/input').rglob('rich-input-manifest.json')); rich=rich_manifest.parent
 for name,expected in json.loads(rich_manifest.read_text()).items():assert hashlib.sha256((rich/name).read_bytes()).hexdigest()==expected
 X=np.load(rich/'rich.npy')
 assert not np.isinf(X).any()
 model=CatBoostRegressor(iterations=1388,depth=5,learning_rate=.05,l2_leaf_reg=10,loss_function='RMSE',random_seed=seed,thread_count=2,verbose=200,allow_writing_files=False,border_count=64,nan_mode='Min')
 model.fit(X[train],target);p=model.predict(X[test]);alpha=0.8999999999999999
 model.save_model(str(output/'model.cbm'))
 replay=CatBoostRegressor();replay.load_model(str(output/'model.cbm'))
 np.testing.assert_allclose(replay.predict(X[test]),p,atol=1e-12,rtol=0)
 recipe=dict(iterations=1388,depth=5,learning_rate=.05,l2_leaf_reg=10,rank_zero_alpha=alpha)
elif MODEL=='cnn64':
 d=dict(np.load(inputs/'features.npz'));ti,ei,pre=_preprocess(d,train,test,'rescnn')
 model=_Regressor('rescnn',ti[0].shape[1],64,.3)
 loader=DataLoader(TensorDataset(*ti,torch.from_numpy(target)),batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(seed))
 optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
 history=[]
 for epoch in range(1,9):
  model.train();total=0.;n=0
  for batch in loader:
   optimizer.zero_grad(set_to_none=True);pred=model(*batch[:-1]);loss=torch.nn.functional.mse_loss(pred,batch[-1]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();total+=loss.item()*len(pred);n+=len(pred)
  row=dict(epoch=epoch,loss=total/n,seconds=time.monotonic()-begun);history.append(row);print(json.dumps(row),flush=True)
 p=_predict(model,ei,torch.device('cpu'),128).astype(np.float64);alpha=.90
 torch.save(dict(state_dict=model.state_dict(),kind='rescnn',width=64,dropout=.3,tab_dim=ti[0].shape[1]),output/'model.pt')
 np.savez_compressed(output/'preprocessing.npz',**pre)
 fresh=_Regressor('rescnn',ti[0].shape[1],64,.3);fresh.load_state_dict(torch.load(output/'model.pt',weights_only=True)['state_dict'])
 np.testing.assert_allclose(_predict(fresh,ei,torch.device('cpu'),128),p,atol=1e-6,rtol=0)
 pd.DataFrame(history).to_csv(output/'epochs.csv',index=False)
 recipe=dict(epochs=8,lr=.001,weight_decay=.001,batch_size=256,width=64,rank_zero_alpha=alpha)
else:raise ValueError(MODEL)
threshold=alpha*float(np.mean(y==0));calibrated=np.where(p<=threshold,0.,p)
assert np.isfinite(calibrated).all() and np.ptp(calibrated)>0
frame=m.loc[test,['gene_name']].copy();frame['gex_predicted']=calibrated
frame.to_csv(output/(MODEL+'.csv'),index=False)
continuous=frame.copy();continuous['gex_predicted']=p;continuous.to_csv(output/(MODEL+'_continuous.csv'),index=False)
roundtrip=pd.read_csv(output/(MODEL+'.csv'))
assert roundtrip.columns.tolist()==['gene_name','gex_predicted'] and roundtrip.gene_name.tolist()==m.loc[test,'gene_name'].tolist()
np.testing.assert_allclose(roundtrip.gex_predicted,calibrated,rtol=1e-14,atol=1e-14)
record=dict(model=MODEL,seed=seed,recipe=recipe,n_train=int(train.sum()),n_predict=int(test.sum()),seconds=time.monotonic()-begun,device='cpu',torch=torch.__version__,python=platform.python_version(),threshold=threshold,input_manifest=manifest,parameters=sum(x.numel() for x in model.parameters()) if MODEL=='cnn64' else None)
(output/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
shutil.make_archive(str(work/(MODEL+'-outputs')),'gztar',output)
print(json.dumps(dict(event='prediction_complete',**record)),flush=True)
