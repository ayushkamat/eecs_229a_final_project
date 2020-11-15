from dotmap import DotMap
from trainers.basic_trainer import BasicTrainer
from data.toy_gauss_data import ToyGauss
from models.mlp import mlp
from torch import nn
from torch.optim import Adam

config = DotMap()

config.trainer = BasicTrainer
config.tp.epochs = 16
config.tp.log_train_every = 1000
config.tp.loss = nn.CrossEntropyLoss()
config.tp.test_loss = nn.CrossEntropyLoss() # train and test separate for flexibility

config.opt = Adam
config.op.lr = 1e-3

config.dataset = ToyGauss
config.dp.seed = 1 # seed must match between test and train in order to use same underlying gaussians
config.dp.gauss_dim = 3
config.dp.num_classes = 3
config.dp.batch_size = 128
config.dp.num_samples = 100000

config.test_dataset = ToyGauss
config.tdp.seed = 1
config.tdp.gauss_dim = 3
config.tdp.num_classes = 3
config.tdp.batch_size = 128
config.tdp.num_samples = 1000

config.model = mlp
config.mp.input_size = config.dp.gauss_dim
config.mp.hidden_sizes = [8, 32, 16]
config.mp.output_size = config.dp.num_classes
config.mp.activation = nn.ReLU()
config.mp.output_activation= nn.Identity()
