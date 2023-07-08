import torch
import pytorch_lightning as pl
import numpy as np
from utils import global_var
global_var._init()  # 全局变量初始化，用于存储实验编号
exp_idx=global_var.set_value('exp_index',0)

import warnings
warnings.filterwarnings("ignore")


def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构

def single_exp(model_name,train_hypers,model_hypers=None):
    same_seeds(0)
    # 初始化模型：
    global Model
    exec('from models.Models import '+model_name+' as Model',globals())
    model=Model(train_hypers) if not model_hypers else Model(train_hypers,model_hypers)

    # 初始化训练器：
    from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
    early_stopping=EarlyStopping('RMSE',patience=train_hypers['patience'])
    filename='exp='+str(global_var.get_value('exp_index'))+'_model='+model_name#+'_{epoch}_{RMSE:.4f}',
    checkpoint=ModelCheckpoint(
        filename=filename,
        monitor='RMSE',
        save_top_k=1,
        mode='min',
        save_last=False)
    
    trainer=pl.Trainer(
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        max_epochs=train_hypers['max_epochs'],
        enable_checkpointing=True,
        callbacks=[checkpoint,early_stopping],
        log_every_n_steps=10
    )
    # 模型训练和测试：
    trainer.fit(model)
    path='lightning_logs/version_'+str(global_var.get_value('exp_index'))+'/checkpoints/'+filename+'.ckpt'
    model=Model.load_from_checkpoint(path,train_hypers=train_hypers) if not model_hypers else Model.load_from_checkpoint(path,train_hypers=train_hypers,model_hypers=model_hypers)
    trainer.test(model)

def batch_exp(batch_model_names,batch_train_hypers,batch_model_hypers,start_version=0,skip=False):
    
    if not skip:
        exp_idx=global_var.set_value('exp_index',start_version-1)
    else:
        exp_idx=global_var.set_value('exp_index',-1)

    for model_name,train_hypers,model_hypers in zip(batch_model_names,batch_train_hypers,batch_model_hypers):
        exp_idx=global_var.get_value('exp_index')
        exp_idx=global_var.set_value('exp_index',exp_idx+1)
        if exp_idx<start_version:continue
        single_exp(model_name,train_hypers,model_hypers)

