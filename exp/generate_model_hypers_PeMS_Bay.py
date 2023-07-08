import torch
import numpy as np
# --------------------------


def gen_model_hypers(model_name,output_len):
    # generate all model hyperparameters in batched experiments.
    input_len=min(output_len*4,108)
    label_len=min(output_len*2,108)
    k={3:2,6:4,12:8,24:12,48:16,96:20}[output_len]
    c={3:4,6:5,12:8,24:15,48:25,96:25}[output_len]
    e_layers={3:1,6:1,12:1,24:1,48:2,96:2}[output_len]
    d_layers={3:1,6:2,12:3,24:3,48:3,96:3}[output_len]
    data_path='data/pems-bay/pems-bay_30days.csv'
    adj_path='data/pems-bay/adj_pems-bay_0.99.csv'
    time_stamp_path='data/pems-bay/time_stamp_30days.csv'
    
    def get_adj(path):
        '''
        this function will skip the first row.
        '''
        
        adj=np.loadtxt(path,delimiter=',',skiprows=1)            # using numpy to read
        adj=torch.from_numpy(adj).to(torch.float32)              # convert to tensor
        return adj
    
    if model_name=='MGCGRU':
        model_hypers={
            'dataset_params':{
                'data_path':data_path,
                'input_len':input_len,
                'output_len':output_len,
            },
            'MGCN_hypers':{
                'adj':get_adj(path=adj_path),
                'k':k
            },
            'GRU_hypers':{
                'batch_first':True,
                'input_size':325,
                'hidden_size':128,
                'num_layers':2
            },
        }
    elif model_name=='GCI':
        model_hypers={
            'informer_hypers':{
                'enc_in':325,  
                'dec_in':325,
                'c_out':325,
                'e_layers':e_layers,
                'd_layers':d_layers,
                'seq_len':input_len,
                'label_len':label_len,
                'out_len':output_len,
                'd_model':128,
                'n_heads':8,
                'd_ff':512,
                'dropout':0.1,
                'freq':'d',
                'embed':'timeF',
                'attn':'prob',
                'factor':c},
            'data_params':{
                'data_path':data_path,
                'time_stamp_path':time_stamp_path,
                'adj_path':adj_path,
            }
        }
    elif model_name=='Informer':
        model_hypers={
            'informer_hypers':{
                'enc_in':325,    
                'dec_in':325,
                'c_out':325,
                'e_layers':e_layers,
                'd_layers':d_layers,
                'seq_len':input_len,
                'label_len':label_len,
                'out_len':output_len,
                'd_model':128,
                'n_heads':8,
                'd_ff':512,
                'dropout':0.1,
                'freq':'d',
                'embed':'timeF',
                'attn':'prob',
                'factor':c},
            'data_params':{
                'data_path':data_path,
                'time_stamp_path':time_stamp_path,
                'adj_path':adj_path,
            }
        }
    elif model_name=='MGCI':
        model_hypers={
            'informer_hypers':{
                'enc_in':325,    
                'dec_in':325,
                'c_out':325,
                'e_layers':e_layers,
                'd_layers':d_layers,
                'seq_len':input_len,
                'label_len':label_len,
                'out_len':output_len,
                'd_model':128,
                'n_heads':8,
                'd_ff':512,
                'dropout':0.1,
                'freq':'d',
                'embed':'timeF',
                'attn':'prob',
                'factor':c},
            'data_params':{
                'data_path':data_path,
                'time_stamp_path':time_stamp_path},
            'mgcn_hypers':{
                'adj':get_adj(path=adj_path),
                'k':k
            }
        }

    return model_hypers
    







