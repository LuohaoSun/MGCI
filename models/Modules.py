import math
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from matplotlib import pyplot as plt
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from torch.utils.data import DataLoader
from torch.utils.tensorboard import summary

from torchmetrics import MeanAbsoluteError
from utils.metrics import RootMeanSquaredError

class TimeSeriesBaseModule(pl.LightningModule):
    '''
    专门用于时间序列模型,集成了打印metrics和绘图的功能.
    args:
    - train_hypers 传入的训练参数,用于生成dataloader和优化器kt
    - unnormalize 需要dataset具有max属性,测试集metrics计算前进行反归一化

    继承后需定义:
    - self.loss_func layer
    - self.train_dataset (具有max属性)
    - self.test_dataset
    - self.validation_dataset
    - forward(self,x) -> output
    如有需要,也可自定义self.train_hypers
    '''
    def __init__(self,unnormalize=True,plot_interval=1) -> None:
        super().__init__()
        
        self.unnormalize=unnormalize
        self.plot_interval=plot_interval

        self.metrics=torch.nn.ModuleDict({
            'MAE':MeanAbsoluteError(),
            'RMSE':RootMeanSquaredError()
        })

        self.epoch=1

    def forward(self, *args, **kwargs):
        return 'hello, world'
    
    '''-------------------------优化器--------------------------'''
    def configure_optimizers(self):
        optimizer=torch.optim.RAdam(self.parameters(),lr=self.train_hypers['lr'])
        # scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=self.train_hypers['step_size'],gamma=self.train_hypers['gamma'],verbose=False)
        return [optimizer]# ,[scheduler]

    '''--------------------dataloaders-------------------------'''
    def train_dataloader(self) :
        return DataLoader(self.train_dataset,self.train_hypers['batch_size'])
    def test_dataloader(self) :
        return DataLoader(self.test_dataset,self.train_hypers['batch_size'])   
    def val_dataloader(self) :
        return DataLoader(self.validation_dataset,self.train_hypers['batch_size'])  

    '''-----------------------trianing-------------------------'''
    def training_step(self,batch,batch_idx) :
        x,y=batch
        output=self(x)
        loss=self.loss_func(y,output)
        return loss
    def training_epoch_end(self, outputs) -> None:
        self.epoch+=1
        return 
    '''-----------------------validation-----------------------'''
    def validation_step(self,batch,batch_idx) :
        x,y=batch
        output=self(x)
        return [y,output]
    def validation_epoch_end(self,outputs):
        y=torch.cat([output[0] for output in outputs],dim=0)    # 合并为(num_samples,seq_len,276)
        y_hat=torch.cat([output[1] for output in outputs],dim=0)
        # 打印和记录metrics：
        if self.unnormalize:
            y=y*self.validation_dataset.max
            y_hat=y_hat*self.validation_dataset.max
        metrics=self.compute_metrics(y,y_hat)
        self.log_dict(metrics)
        if self.epoch % self.plot_interval ==0:
            self.plot_node(y,y_hat)
        return metrics
    '''---------------------------test-------------------------'''
    def test_step(self,batch,batch_idx):
        x,y=batch
        output=self(x)
        return [y,output]
    def test_epoch_end(self, outputs):
        # according to test_step, outputs 是 batch num 个 (truth,output)
        y=torch.cat([output[0] for output in outputs],dim=0)    # batch num * (batch_size,...) -> (num_samples,...)
        y_hat=torch.cat([output[1] for output in outputs],dim=0)
        if self.unnormalize:
            y=y*self.test_dataset.max
            y_hat=y_hat*self.test_dataset.max
        # 打印metrics
        from utils import global_var
        path='lightning_logs/version_'+str(global_var.get_value('exp_index'))+'/metrics.txt'
        with open(path,'w') as f:
            f.write(str(self.compute_metrics(y,y_hat)))
            f.close()
        return 
    '''---------------------------utils------------------------'''
    def compute_metrics(self,y,y_hat):
        '''
        计算所有metrics  
        '''
        metrics={}
        for m in self.metrics:
            metrics[m]=self.metrics[m](y_hat,y)
        return metrics
    def plot_output(self,y,y_hat):
        '''
        绘制所有节点序列图像
        '''
        for i in range(y.shape[2]):
            x_points=[i for i in range(y.shape[0])]
            y1_points=y[:,-1,i].cpu().detach().numpy()       # -1 表示,如果预测长度大于1,只绘制最后一个预测时间步
            y2_points=y_hat[:,-1,i].cpu().detach().numpy()
            plt.plot(x_points,y1_points,x_points,y2_points,scalex=300)
            plt.show()
            if i>=7: break
        return
    def plot_node(self,y,y_hat):
        '''
        绘制特定节点时间序列图像
        '''
        x_points=[i for i in range(y.shape[1])]
        y1_points=y[0,:,0].cpu().detach().numpy()       # the first 0 denotes the 1st window of val set, the second 0 denotes the 1st node of graph
        y2_points=y_hat[0,:,0].cpu().detach().numpy()
        plt.plot(x_points,y1_points,x_points,y2_points,scalex=300)
        plt.savefig('images/epoch='+str(self.epoch)+'.png')
        plt.close()
        return

    def get_adj(self,path):
        '''
        会跳过第一行
        '''
        import numpy as np
        adj=np.loadtxt(path,delimiter=',',skiprows=1)            # 使用numpy读取邻接矩阵
        adj=torch.from_numpy(adj).to(torch.float32).to(self.device)               # 转换为张量
        return adj

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L,device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2,1).contiguous(), attn

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm3(x+y)

class Decoder(nn.Module):

    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        pe.require_grad = False

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model)
        w.require_grad = False

        position = torch.arange(0, c_in).unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):# 这里输入实际上是mark
        x = x.long()
        
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    # 注意这里u
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x): # mark actually
        return self.embed(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1,timeFdim=3):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        
        return self.dropout(x)

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1,2)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm2(x+y), attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class EncoderStack(nn.Module):
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens                # list(range(len(en_layer))), 0,1,2...

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        x_stack = []; attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1]//(2**i_len)
            x_s, attn = encoder(x[:, -inp_len:, :]) # 每次取后1/2^i_len, i_len=0,1,2...
            x_stack.append(x_s); attns.append(attn)
        x_stack = torch.cat(x_stack, -2)
        
        return x_stack, attns

class Informer(nn.Module):
    '''
    ### args:
    - enc_in: encoder input dim
    - dec_in: decoder input dim
    - c_out: output dim
    - seq_len: Input sequence length of Informer encoder
    - label_len: Start token length of Informer decoder
    - out_len: prediction seq lenth
    - factor: probsparse attn factor
    - d_model: Dimension of model (defaults to 512)
    - more
    ### input:
    - x_enc: (b_size, seq_len, enc_in)
    - x_mark_enc: (b_size, seq_len,time_dim)
    - x_dec: (b_size, label_len+pred_len , dec_in)
    - x_mark_dec: (b_size, label_len+pred_len, dec_in)
    - more
    ### output:
    - dec_out: (b_size, pred_len, c_out)
    ### where:
    - enc_in = [0:seq_len]
    - dec_in = concat(x_token,[0,])
        - x_token = [seq_len-label_len:seq_len]
        - [0,] = [seq_len:seq_len+pred_len]
    - c_out = [seq_len:seq_len+pred_len]
    ### note:
    - seq_len & label_len 未被使用，因为在进行embedding时不需要知道序列长度
    - end_conv在源代码中本来就是被注释掉的，不知道干嘛的。这两行需要上述两个参数
    '''
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                # modified:--------------------------------------------
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                # modified:--------------------------------------------
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]

class GraphConv(nn.Module):
    '''
    ## args:
        - adj(not A_hat)
        - input_dim = 1
        - output_dim = 1

    ## inputs:
        - (batch_size,num_nodes,input_dim)
    ## outputs:
        - (batch_size,num_nodes,output_dim)
    '''
    def __init__(self,adj,input_dim=1,output_dim=1) -> None:
        super().__init__()

        self.DAD = self.calculate_laplacian_with_self_loop(adj)
        self.num_nodes=self.DAD.shape[0]
        self.input_dim=input_dim
        self.output_dim=output_dim

        self.weights = nn.Parameter(
            torch.FloatTensor(self.input_dim,self.output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self.output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, 0)

    def forward(self,x):
        self.DAD=self.DAD.to(x.device)
        x=torch.matmul(self.DAD,x)      # (156,156) broadcast @ (batch_size,156,1) 
        x=torch.matmul(x,self.weights)  # (batch_size,156,1) @ (1,1) broadcast
        x=x+self.biases                 # (batch_size,156,1) + (1) broadcast
        return x

    def calculate_laplacian_with_self_loop(self,matrix):
        matrix=matrix# .to(torch.device('cpu'))       # 这个函数不兼容mps设备
        if matrix[0,0]==0:                           # 判断是否存在自环,没有的话加上
            matrix = matrix + torch.eye(matrix.size(0))
        row_sum = matrix.sum(1)
        d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        normalized_laplacian = (
            matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
        )
        return normalized_laplacian
    
class MGCN(nn.Module):
    '''
    input shape: (*, num_nodes)
    hidden shape: (*, num_nodes, k)
    output shape: (*, num_nodes)
    '''
    def __init__(self,adj,k) -> None:
        super().__init__()
        self.k=k
        distance=self.distance_matrix(adj)
        self.gcns=nn.ModuleList()    # 生成k个gcn，分别使用k阶邻接矩阵.使用ModuleList注册。
        for i in range(1,k+1):  
            i_adj=self.k_adj_matrix(distance,i)
            i_gcn=GraphConv(adj=i_adj,input_dim=1,output_dim=1)
            self.gcns.append(i_gcn)
        self.fusion=nn.Linear(in_features=k,out_features=1,bias=False)
    def forward(self,x):
        x=x.unsqueeze(dim=-1)       # (*, num_nodes) -> (*, num_nodes, 1)
        kx=[]
        for gcn in self.gcns:
            ix=gcn(x)               # (*, num_nodes, 1) -> (*, num_nodes, 1)
            kx.append(ix)
        kx=torch.cat(kx,dim=-1)     # [(*, num_nodes, 1) * k] -> (*, num_nodes, k)
        x=self.fusion(kx)               # (*, num_nodes, k) -> (*, num_nodes, 1)
        x=x.squeeze(dim=-1)       # (*, num_nodes, 1) -> (*, num_nodes)
        return x
        
    def distance_matrix(self,adj_matrix):
        '''
        # 根据邻接矩阵计算距离矩阵
        '''
        n = len(adj_matrix)
        distance_matrix = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    distance_matrix[i][j] = 0
                elif adj_matrix[i][j] == 1:
                    distance_matrix[i][j] = 1
                else:
                    distance_matrix[i][j] = float('inf')
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    distance_matrix[i][j] = min(distance_matrix[i][j], distance_matrix[i][k] + distance_matrix[k][j])
        return distance_matrix
    

    def k_adj_matrix(self,distance_matrix, k):
        '''
        #根据距离矩阵计算k阶邻接矩阵
        #注意: 仅保留距离恰好为k时的节点
        '''
        n = len(distance_matrix)
        adj_matrix = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if distance_matrix[i][j] == k:  # 只有第i个点和第j个点之间距离恰好为k时，邻接矩阵中对应位置才会赋值为1
                    adj_matrix[i][j] = 1
        adj_matrix=torch.from_numpy(adj_matrix).to(torch.float32)
        return adj_matrix
