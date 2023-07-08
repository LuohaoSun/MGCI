import data.Datasets as Datasets
import models.Modules as Modules
import torch
import torch.nn as nn

class MGCLSTM(Modules.TimeSeriesBaseModule):
    '''
    ### args:
    - train_hypers, see Modules.TimeSeriesBaseModule
    '''
    def __init__(self,train_hypers,model_hypers) -> None:
        super().__init__()
        self.train_hypers=train_hypers

        self.model_hypers={
            'dataset_params':{
            'data_path':'data/BJinflow/in_10min_trans.csv',
            'input_len':48,
            'output_len':12,
            },
            'MGCN_hypers':{
                'adj':self.get_adj(path='data/BJinflow/adjacency_with_label.csv'),
                'k':1
            },
            'LSTM_hypers':{
            'batch_first':True,
            'input_size':276,
            'hidden_size':128,
            'num_layers':2
            },
        } if not model_hypers else model_hypers

        
        
        self.dataset_params=self.model_hypers['dataset_params']
        self.MGCN_hypers=self.model_hypers['MGCN_hypers']
        self.LSTM_hypers=self.model_hypers['LSTM_hypers']
        
        self.train_dataset=Datasets.TimeSeriesDataset(**self.dataset_params,flag='training')
        self.validation_dataset=Datasets.TimeSeriesDataset(**self.dataset_params,flag='validation')
        self.test_dataset=Datasets.TimeSeriesDataset(**self.dataset_params,flag='test')

        self.mgcn=Modules.MGCN(**self.MGCN_hypers)
        self.lstm=nn.LSTM(**self.LSTM_hypers)
        self.fc=nn.Linear(self.LSTM_hypers['hidden_size'],self.LSTM_hypers['input_size'])
        self.loss_func=nn.MSELoss()

    def forward(self,x):
        x=self.mgcn(x)
        x,(h,c)=self.lstm(x)    # (b_size, seq_len, dim)
        x=x[:,-1,:]             # (b_size, dim)
        x=self.fc(x)            # (b_size, dim)
        x=x.unsqueeze(dim=1)    # (b_size, 1, dim)
        output=[x]
        for step in range(self.dataset_params['output_len']-1):   # 自回归预测多步
            x,(h,c)=self.lstm(x,(h,c))
            x=x[:,-1,:]
            x=self.fc(x)
            x=x.unsqueeze(dim=1)    # (b_size, 1, dim)
            output.append(x)
        output=torch.cat(output,dim=1)  # (b_size, output_len, dim)
        return output
    
class MGCGRU(Modules.TimeSeriesBaseModule):
    '''
    ### args:
    - train_hypers, see Modules.TimeSeriesBaseModule
    '''
    def __init__(self,train_hypers,model_hypers) -> None:
        super().__init__()
        self.train_hypers=train_hypers

        self.model_hypers={
            'dataset_params':{
                'data_path':'data/BJinflow/in_10min_trans.csv',
                'input_len':48,
                'output_len':12,
            },
            'MGCN_hypers':{
                'adj':self.get_adj(path='data/BJinflow/adjacency_with_label.csv'),
                'k':1
            },
            'GRU_hypers':{
                'batch_first':True,
                'input_size':276,
                'hidden_size':128,
                'num_layers':2
            },
        } if not model_hypers else model_hypers

        
        self.dataset_params=self.model_hypers['dataset_params']
        self.MGCN_hypers=self.model_hypers['MGCN_hypers']
        self.GRU_hypers=self.model_hypers['GRU_hypers']
        
        self.train_dataset=Datasets.TimeSeriesDataset(**self.dataset_params,flag='training')
        self.validation_dataset=Datasets.TimeSeriesDataset(**self.dataset_params,flag='validation')
        self.test_dataset=Datasets.TimeSeriesDataset(**self.dataset_params,flag='test')

        self.mgcn=Modules.MGCN(**self.MGCN_hypers)
        self.gru=nn.GRU(**self.GRU_hypers)
        self.fc=nn.Linear(self.GRU_hypers['hidden_size'],self.GRU_hypers['input_size'])
        self.loss_func=nn.MSELoss()

    def forward(self,x):
        x=self.mgcn(x)
        x,h=self.gru(x)    # (b_size, seq_len, dim)
        x=x[:,-1,:]             # (b_size, dim)
        x=self.fc(x)            # (b_size, dim)
        x=x.unsqueeze(dim=1)    # (b_size, 1, dim)
        output=[x]
        for step in range(self.dataset_params['output_len']-1):   # 自回归预测多步
            x,h=self.gru(x,h)
            x=x[:,-1,:]
            x=self.fc(x)
            x=x.unsqueeze(dim=1)    # (b_size, 1, dim)
            output.append(x)
        output=torch.cat(output,dim=1)  # (b_size, output_len, dim)
        return output

class Informer(Modules.TimeSeriesBaseModule):
    def __init__(self,train_hypers,model_hypers=None):
        super().__init__()
        self.train_hypers=train_hypers  
        self.model_hypers= {
            'informer_hypers':{
                'enc_in':276,            # 时间序列的dim,即于图的节点数
                'dec_in':276,
                'c_out':276,
                'e_layers':1,
                'd_layers':2,
                'seq_len':48,
                'label_len':48,
                'out_len':12,
                'd_model':128,
                'n_heads':8,
                'd_ff':512,
                'dropout':0.1,
                'freq':'d',
                'embed':'timeF',
                'attn':'prob',
                'factor':5},
            'data_params':{
                'data_path':'data/BJinflow/in_10min_trans.csv',
                'time_stamp_path':'data/BJinflow/time_stamp.csv',
                'adj_path':'data/BJinflow/adjacency_with_label.csv',
            }
        }if not model_hypers else model_hypers

        self.informer_hypers=self.model_hypers['informer_hypers']
        self.data_params=self.model_hypers['data_params']

        
        self.train_dataset=Datasets.InformerDataset(
            time_series_path=self.data_params['data_path'],
            time_stamp_path=self.data_params['time_stamp_path'],
            seq_len=self.informer_hypers['seq_len'],
            label_len=self.informer_hypers['label_len'],
            pred_len=self.informer_hypers['out_len'],
            flag='training'
        )
        self.test_dataset=Datasets.InformerDataset(
            time_series_path=self.data_params['data_path'],
            time_stamp_path=self.data_params['time_stamp_path'],
            seq_len=self.informer_hypers['seq_len'],
            label_len=self.informer_hypers['label_len'],
            pred_len=self.informer_hypers['out_len'],
            flag='test'
        )
        self.validation_dataset=Datasets.InformerDataset(
            time_series_path=self.data_params['data_path'],
            time_stamp_path=self.data_params['time_stamp_path'],
            seq_len=self.informer_hypers['seq_len'],
            label_len=self.informer_hypers['label_len'],
            pred_len=self.informer_hypers['out_len'],
            flag='validation'
        )

        self.informer=Modules.Informer(**self.informer_hypers)
        self.loss_func=nn.MSELoss()

    def forward(self,inputs):
        en_in,en_mark,de_in,de_mark=inputs
        output=self.informer(en_in,en_mark,de_in,de_mark)
        return output

class GCI(Modules.TimeSeriesBaseModule):
    def __init__(self,train_hypers,model_hypers=None):
        super().__init__()
        self.train_hypers=train_hypers  
        self.model_hypers= {
            'informer_hypers':{
                'enc_in':276,            # 时间序列的dim,即于图的节点数
                'dec_in':276,
                'c_out':276,
                'e_layers':1,
                'd_layers':2,
                'seq_len':48,
                'label_len':48,
                'out_len':12,
                'd_model':128,
                'n_heads':8,
                'd_ff':512,
                'dropout':0.1,
                'freq':'d',
                'embed':'timeF',
                'attn':'prob',
                'factor':5},
            'data_params':{
                'data_path':'data/BJinflow/in_10min_trans.csv',
                'time_stamp_path':'data/BJinflow/time_stamp.csv',
                'adj_path':'data/BJinflow/adjacency_with_label.csv',
            }
        }if not model_hypers else model_hypers

        self.informer_hypers=self.model_hypers['informer_hypers']
        self.data_params=self.model_hypers['data_params']

        
        self.train_dataset=Datasets.InformerDataset(
            time_series_path=self.data_params['data_path'],
            time_stamp_path=self.data_params['time_stamp_path'],
            seq_len=self.informer_hypers['seq_len'],
            label_len=self.informer_hypers['label_len'],
            pred_len=self.informer_hypers['out_len'],
            flag='training'
        )
        self.test_dataset=Datasets.InformerDataset(
            time_series_path=self.data_params['data_path'],
            time_stamp_path=self.data_params['time_stamp_path'],
            seq_len=self.informer_hypers['seq_len'],
            label_len=self.informer_hypers['label_len'],
            pred_len=self.informer_hypers['out_len'],
            flag='test'
        )
        self.validation_dataset=Datasets.InformerDataset(
            time_series_path=self.data_params['data_path'],
            time_stamp_path=self.data_params['time_stamp_path'],
            seq_len=self.informer_hypers['seq_len'],
            label_len=self.informer_hypers['label_len'],
            pred_len=self.informer_hypers['out_len'],
            flag='validation'
        )
        self.gcn1=Modules.GraphConv(adj=self.get_adj(path=self.data_params['adj_path']))
        self.gcn2=Modules.GraphConv(adj=self.get_adj(path=self.data_params['adj_path']))

        self.informer=Modules.Informer(**self.informer_hypers)
        self.loss_func=nn.MSELoss()

    def forward(self,inputs):
        en_in,en_mark,de_in,de_mark=inputs
        en_in=en_in.unsqueeze(dim=-1)
        de_in=de_in.unsqueeze(dim=-1)
        en_in=self.gcn1(en_in)
        de_in=self.gcn2(de_in)
        en_in=en_in.squeeze(dim=-1)
        de_in=de_in.squeeze(dim=-1)
        output=self.informer(en_in,en_mark,de_in,de_mark)
        return output

class MGCI(Modules.TimeSeriesBaseModule):
    def __init__(self,train_hypers,model_hypers=None):
        super().__init__()
        self.train_hypers=train_hypers  
        self.model_hypers= {
            'informer_hypers':{
                'enc_in':276,            # 时间序列的dim,即于图的节点数
                'dec_in':276,
                'c_out':276,
                'e_layers':1,
                'd_layers':2,
                'seq_len':48,
                'label_len':48,
                'out_len':12,
                'd_model':128,
                'n_heads':8,
                'd_ff':512,
                'dropout':0.1,
                'freq':'d',
                'embed':'timeF',
                'attn':'prob',
                'factor':5},
            'data_params':{
                'data_path':'data/BJinflow/in_10min_trans.csv',
                'time_stamp_path':'data/BJinflow/time_stamp.csv'},
            'mgcn_hypers':{
                'adj':self.get_adj(path='data/BJinflow/adjacency_with_label.csv'),
                'k':1
            }
        }if not model_hypers else model_hypers

        self.informer_hypers=self.model_hypers['informer_hypers']
        self.data_params=self.model_hypers['data_params']
        self.mgcn_hypers=self.model_hypers['mgcn_hypers']

        
        self.train_dataset=Datasets.InformerDataset(
            time_series_path=self.data_params['data_path'],
            time_stamp_path=self.data_params['time_stamp_path'],
            seq_len=self.informer_hypers['seq_len'],
            label_len=self.informer_hypers['label_len'],
            pred_len=self.informer_hypers['out_len'],
            flag='training'
        )
        self.test_dataset=Datasets.InformerDataset(
            time_series_path=self.data_params['data_path'],
            time_stamp_path=self.data_params['time_stamp_path'],
            seq_len=self.informer_hypers['seq_len'],
            label_len=self.informer_hypers['label_len'],
            pred_len=self.informer_hypers['out_len'],
            flag='test'
        )
        self.validation_dataset=Datasets.InformerDataset(
            time_series_path=self.data_params['data_path'],
            time_stamp_path=self.data_params['time_stamp_path'],
            seq_len=self.informer_hypers['seq_len'],
            label_len=self.informer_hypers['label_len'],
            pred_len=self.informer_hypers['out_len'],
            flag='validation'
        )
        self.mgcn1=Modules.MGCN(**self.model_hypers['mgcn_hypers'])
        self.mgcn2=Modules.MGCN(**self.model_hypers['mgcn_hypers'])

        self.informer=Modules.Informer(**self.informer_hypers)
        self.loss_func=nn.MSELoss()

    def forward(self,inputs):
        en_in,en_mark,de_in,de_mark=inputs
        en_in=self.mgcn1(en_in)
        de_in=self.mgcn2(de_in)
        output=self.informer(en_in,en_mark,de_in,de_mark)
        return output
    
