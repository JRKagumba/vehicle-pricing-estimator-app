import pickle
import numpy as np

import torch
import torch.nn as nn

class TabularModel(nn.Module):
    def __init__(self, emb_szs, n_cont, out_sz, layers, p=0.5):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        layerlist = []
        n_emb = sum((nf for ni,nf in emb_szs))
        n_in = n_emb + n_cont
        for i in layers:
            layerlist.append(nn.Linear(n_in,i)) 
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1],out_sz))
        self.layers = nn.Sequential(*layerlist)
    def forward(self, x_cat, x_cont):
        embeddings = []
        for i,e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:,i]))
        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        return self.layers(x)


# LOAD TRANSFORMATIONS

transform_load_path='data/vehicle_analysis_transform.pkl'

with open(transform_load_path, 'rb') as handle:
    loaded_transform_dict = pickle.load(handle)

# LOAD PARAMETERS & INSTANTIATE
params_load_path=f'data/6191_vehicle_analysis_params.pkl'

with open(params_load_path, 'rb') as handle:
    loaded_params_dict = pickle.load(handle)

reg_model = TabularModel(
    loaded_params_dict['emb_szs'],
    loaded_params_dict['n_cont'],
    loaded_params_dict['out_sz'],
    loaded_params_dict['layers'],
    loaded_params_dict['p']
)

# LOAD MODEL
model_load_path=f'data/6191_vehicle_analysis_model.pt'

reg_model.load_state_dict(torch.load(model_load_path));
reg_model.eval() # be sure to run this step!

def make_predictions(values_list):

    make=values_list[0]
    model=values_list[1]
    year=int(values_list[2])
    mileage=float(values_list[3])
    
    ## TRANSFORM CAT VARIABLES
    make_trnsfm = loaded_transform_dict['Vehicle_Make'][make]
    model_trnsfm = loaded_transform_dict['Vehicle_Model'][model]

    xcats = np.stack([[make_trnsfm],[model_trnsfm]], 1)
    xcats = torch.tensor(xcats, dtype=torch.int64)
    xconts = np.stack([[year],[mileage]], 1)
    xconts = torch.tensor(xconts, dtype=torch.float)

    with torch.no_grad():
        prediction = reg_model(xcats, xconts)

    return(round(prediction.item(),2))