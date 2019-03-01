from scipy.io import arff
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pdb
import torch.nn as nn
import torch.nn.functional as F
import torch
import fastai.basic_data
import fastai.basic_train
import fastai.train
from fastai.metrics import accuracy
import warnings

def load_df(task):
    data = arff.loadarff('NewTSCProblems/%s/%s_TRAIN.arff'%(task,task))
    df = pd.DataFrame(data[0])
    return df

def cleanup(df):
    df.columns = [k for k in range(df.shape[1]-1)]+['target']
    for k in df.columns[:-1]:
        df[k] = df[k].astype('float')
    if df.target.dtype == 'object':
        df['target'] = df['target'].apply(lambda x: x.decode('ascii')).astype('int')
    if sorted(df.target.unique()) != list(np.arange(df.target.nunique())):
        new_targs = pd.DataFrame({'target':df.target.unique()}).reset_index()
        df = pd.merge(df, new_targs, left_on='target', right_on='target').drop('target',axis=1).rename(columns={'index':'target'})
    ts = pd.melt(df.reset_index(), id_vars=['index','target'], var_name='time').rename(columns={'index':'id'})
    ts = ts.groupby(['id','time','target']).value.mean().reset_index()
    df = df.sample(df.shape[0], replace=False)
    return df, ts

def fetch_predictions(model, ts, df, output=False):
    data = partial_lm_dataset(df)
    date_range = [k for k in range(1,df.shape[1]-1)]
    dl = DataLoader(data, batch_size=32, shuffle=False)
    pieces = []
    model.cpu()
    model.eval()
    for x,y in dl:
        pieces.append(model(x).detach().numpy())
    pct = pd.DataFrame(np.concatenate(pieces).squeeze())
    pct.index = df.index
    pct = pct.transpose()
    pct['time'] = date_range
    pts = pd.melt(pct, id_vars='time', var_name = 'id', value_name = 'predicted_value')
    pts.id = pts.id.astype('int')
    pts = pd.merge(ts, pts, left_on=['time','id'], right_on=['time', 'id'], how='left').fillna(0)
    return pts

def graph_ts(ts):
    for k in ts.target.unique():
        fig, axes = plt.subplots(figsize=(15,5))
        sns.tsplot(ts[ts.target == k], time='time', unit='id', condition='target', value='value', err_style='unit_traces', ax=axes)    
    fig, axes = plt.subplots(figsize=(15,5))
    sns.tsplot(ts, time='time', unit='id', condition='target', value='value', err_style='unit_traces', ax=axes)
    return None

def graph_predictions(pts, size=5):
    ids = np.random.choice(np.arange(pts.id.nunique()), size=size)
    for k in ids:
        piece = pts[pts.id == pts.id.unique()[k]]
        piece.index = piece.time
        fig,ax = plt.subplots()
        piece.value.plot.line()
        piece.predicted_value.plot.line()
        ax.legend()
    return None

def get_cm(clf, val_dl):
    clf.data.valid_dl = val_dl
    x, y = clf.get_preds()
    preds = torch.max(x, dim=1)[1]
    cm = confusion_matrix(y, preds)
    return cm

class partial_lm_dataset(Dataset):
    def __init__(self, ts):
        self.x = torch.stack([torch.Tensor(ts.iloc[k][:-1].values.astype('float')) for k in range(ts.shape[0])], dim=1)
            
    def __len__(self):
        return self.x.size(1)
    
    def __getitem__(self, idx):
        return self.x[:-1,idx].unsqueeze(1), self.x[1:,idx].unsqueeze(1)