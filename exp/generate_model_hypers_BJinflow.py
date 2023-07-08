import torch
# --------------------------

def gen_model_hypers(model_name,output_len):

    input_len=min(output_len*4,108)
    label_len=min(output_len*2,108)
    k={3:3,6:6,12:15,24:15,48:15,96:15}[output_len]
    c={3:3,6:5,12:15,24:15,48:20,96:20}[output_len]
    e_layers={3:1,6:1,12:1,24:1,48:2,96:2}[output_len]
    d_layers={3:1,6:3,12:2,24:2,48:2,96:2}[output_len]
    data_path='data/BJinflow/in_10min_trans.csv'
    adj_path='data/BJinflow/adjacency_with_label.csv'
    time_stamp_path='data/BJinflow/time_stamp.csv'
    
    def get_adj(path):
        import numpy as np
        adj=np.loadtxt(path,delimiter=',',skiprows=1)        
        adj=torch.from_numpy(adj).to(torch.float32)    
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
                'input_size':276,
                'hidden_size':128,
                'num_layers':2
            },
        }
    elif model_name=='GCI':
        model_hypers={
            'informer_hypers':{
                'enc_in':276,   
                'dec_in':276,
                'c_out':276,
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
                'enc_in':276,  
                'dec_in':276,
                'c_out':276,
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
                'enc_in':276,            # 时间序列的dim,即于图的节点数
                'dec_in':276,
                'c_out':276,
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
    






