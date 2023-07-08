from exp.exp import *
import argparse

parser = argparse.ArgumentParser(description='Train, eval and test the MGCI model. Change the model hyper-parameters in exp/ for detailed fine-tuning')
parser.add_argument('--b_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--gamma', type=float, default=1, metavar='M',
                    help='Learning rate step gamma (default: 1)')
parser.add_argument('--s_size', type=int, default=1, metavar='N',
                    help='Learning rate step size (default: 1)')
parser.add_argument('--max_epochs', type=int, default=2000, metavar='N',
                    help='number of epochs to train (default: 2000)')
parser.add_argument('--patience', type=int, default=200, metavar='N',
                    help='Early stopping patience (default: 200)')
parser.add_argument('--dataset', type=str, default='metrobj', choices=['metrobj', 'pems_bay'],
                    help='Dataset to use (default: "metrobj")')

args = parser.parse_args()

dataset = args.dataset
if dataset == 'metrobj':
    from exp.generate_model_hypers_BJinflow import gen_model_hypers
elif dataset == 'pems_bay':
    from exp.generate_model_hypers_PeMS_Bay import gen_model_hypers
else:
    raise ValueError('Invalid dataset argument, must be either metrobj or pemsbay, got: %s' % dataset)

batch_train_hypers = [{
    'batch_size': args.b_size,
    'lr': args.lr,
    'gamma': args.gamma,
    'step_size': args.s_size,
    'max_epochs': args.max_epochs,
    'patience': args.patience, 
}] * 6

batch_model_names=['MGCI']*6

batch_model_hypers=[
    gen_model_hypers('MGCI', output_len) for output_len in [3,6,12,24,48,96]
    ]


# run training
batch_exp(batch_model_names, batch_train_hypers, batch_model_hypers)