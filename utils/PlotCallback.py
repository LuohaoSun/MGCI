import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from matplotlib import pyplot as plt

class PlotCallback(Callback):
    def __init__(self):
        super().__init__()
        self.step = 0

    def on_epoch_end(self, trainer, pl_module):
        self.step += 1
        if self.step % 100 == 0:
            # 绘制图表的代码
            # 例如：
            # plt.plot(loss_values)
            # plt.show()
            pass

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return super().on_validation_epoch_end(trainer, pl_module)

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
        y1_points=y[:,:,0].cpu().detach().numpy()       # 0 denotes the 1st node of graph
        y2_points=y_hat[:,:,0].cpu().detach().numpy()
        plt.plot(x_points,y1_points,x_points,y2_points,scalex=300)
        plt.show()
        plt.savefig('images/epoch='+str(self.epoch)+'.png')
        return