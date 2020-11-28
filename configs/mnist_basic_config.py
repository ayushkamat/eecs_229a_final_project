from dotmap import DotMap
from trainers.basic_trainer import BasicTrainer
from data.mnist_data import MNISTData
from models.mlp import mlp
from torch import nn
from torch.optim import Adam
import torch

config = DotMap()
config.seed = 1

config.trainer = BasicTrainer
config.tp.epochs = 16
config.tp.log_train_every = 1000
config.tp.loss = nn.CrossEntropyLoss()
config.tp.test_loss = nn.CrossEntropyLoss() # train and test separate for flexibility
config.tp.device = torch.device('cuda') if config.tp.use_gpu else torch.device('cpu')
config.opt = Adam
config.op.lr = 1e-3

config.dataset = MNISTData
config.dp.device = config.tp.device
config.dp.seed = config.seed # seed must match between test and train in order to use same underlying gaussians
config.dp.classes = [0, 1, 2, 3, 4, 5]
config.dp.resolution = (28, 28)
config.dp.dir = './data/cache/mnist/'
config.dp.num_classes = 10
config.dp.batch_size = 128


config.test_dataset = MNISTData
config.tdp.device = config.tp.device
config.tdp.seed = config.seed
config.tdp.classes = config.dp.classes
config.tdp.resolution = config.dp.resolution
config.tdp.dir = config.dp.dir
config.tdp.num_classes = config.dp.num_classes
config.tdp.batch_size = 128

config.model = mlp
config.mp.device = config.tp.device
config.mp.input_size = config.dp.resolution[0] * config.dp.resolution[1]
config.mp.hidden_sizes = [256, 256, 256]
config.mp.output_size = config.dp.num_classes
config.mp.activation = nn.ReLU()
config.mp.output_activation= nn.Identity()
