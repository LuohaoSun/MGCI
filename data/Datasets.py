import torch
import numpy as np
from torch.utils.data import Dataset

    
class TimeSeriesDataset(Dataset):
    '''
    ### 说明
    可以将.csv文件读取为时间序列dataset,格式要求:
    - 列是时间序列
    - 行是一个时间步的所有节点
    - 第一行是label
    
    ### args
    5. data_path:       file path (for example: data_path = './data/sz_speed.csv')
    1. seq_len:         x_seq_len
    2. pred_len:        y_seq_len
    3. flag
    4. division
    5. normalized:      是否归一化到最大值为1                                         (default=True)  
    
    ### index output = x,y
    1. x:               shape = (seq_len, feature_dim)
    2. y:               shape = (pred_len, feature_dim)
    '''
    def __init__(self,data_path,input_len:int,output_len:int,flag='training',division=[6,2,2],normalized=True,**kw) -> None:
        super().__init__()

        
        '''------导入数据--------'''
        self.data=np.loadtxt(data_path,delimiter=',',skiprows=1)            # 使用numpy读取数据集,跳过第1行
        if normalized:
            self.max=self.data.max()                                            # 存储max值,方便反归一化
            self.data=self.data/self.max                                        # 归一化
        self.data=torch.from_numpy(self.data).to(torch.float32)             # 转换为张量
        
        '''----计算数据集大小----'''
        # 如果样本数不能被划分整除,最后一个时刻数据会被忽略
        training_ratio=division[0]/sum(division)
        validation_ratio=division[1]/sum(division)
        test_ratio=division[2]/sum(division)
        data_len=self.data.shape[0]-input_len-output_len
        train_set_len=int(data_len*training_ratio)      # 计算训练集样本数
        validation_set_len=int(data_len*validation_ratio)
        test_set_len=int(data_len*test_ratio)            # 计算测试集样本数
        
        if train_set_len+validation_set_len+test_set_len<data_len:
            pass
            '''
            print('warning: some samples left')
            print(train_set_len+validation_set_len+test_set_len)
            print(data_len)
            # raise RuntimeError()
            '''
        self.samples=[]
        if flag=='training':                                                       # 训练集信息
            self.lenth=train_set_len                                        # 训练集长度
            start_idx=0                                                     # 第一个样本左侧索引
        elif flag=='validation':                                            
            self.lenth=validation_set_len
            start_idx=train_set_len                                         # 第一个样本左侧索引
        elif flag=='test':
            self.lenth=test_set_len
            start_idx=train_set_len+validation_set_len
        else:
            raise RuntimeError('flag expected training, validation or test, but got '+str(flag))

        '''-----生成所有样本-----'''
        for i in range(self.lenth): 
            i+=start_idx
            sample_x=self.data[i:i+input_len,:]
            sample_y=self.data[i+input_len:i+input_len+output_len,:]
            self.samples.append([sample_x,sample_y])

    def __getitem__(self,index):
        return self.samples[index]

    def __len__(self):
        return self.lenth

class InformerDataset(Dataset):
    '''
    ### 说明

    可以将两个.csv文件读取为时间序列dataset,要求:
    - 第一个.csv文件是时间序列数据
    - 第二个.csv文件是时间戳
    - 两个文件需要在时间上对齐
    - .csv格式要求见sequential_dataset

    ### index output:   x,y
    - x = (en_in,en_mark,de_in,de_mark)
        - en_in:    (seq_len,feature_dim)
        - en_mark:  (seq_len,timefeatrue_dim=3) 
        - de_in:    (label_len+pred_len,feature_dim)
        - de_mark:  (label_len+pred_len,timefeatrue_dim=3)
    - y : (pred_len,feature_dim)
    - where:
        - timefeature=[weekday,hour,minute]    
        - minute = 1,2,...,hour_interval
    '''
    def __init__(self,time_series_path,time_stamp_path,seq_len,label_len,pred_len,flag='training',division=[6,2,2]) -> None:
        super().__init__()
        
        self.label_len=label_len    
        
        
        self.series_set=TimeSeriesDataset(time_series_path,seq_len,pred_len,flag,division)
        self.stamp_set=TimeSeriesDataset(time_stamp_path,seq_len,pred_len,flag,division,normalized=False)
        self.max=self.series_set.max
        self.samples=[]
        for index in range(len(self)):
            # 计算输入：
            en_in,y=self.series_set[index]
            de_in=torch.cat(
                [   en_in[en_in.shape[0]-self.label_len:,:],
                    torch.zeros(*y.shape)
                ],
                dim=0
            ) 

            # 计算stamp：
            en_mark,pred_mark=self.stamp_set[index]
            de_mark=torch.cat(
                [   en_mark[en_mark.shape[0]-self.label_len:,:],
                    pred_mark
                ],  # 拼接en_mark的最后和y_mark
                dim=0
            )

            x=(en_in,en_mark,de_in,de_mark)
            self.samples.append((x,y))

    def __getitem__(self, index):
        return self.samples[index]
    
    def __len__(self):
        
        return len(self.series_set)

















