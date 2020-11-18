from dotmap import DotMap
from trainers.distance_trainer import DistanceTrainer
from data.gmm_data import GMMData, GMMTeacherData
from models.mlp import mlp
from torch import nn
from torch.optim import Adam
import torch

config = DotMap()
config.seed = 1

config.trainer = DistanceTrainer
config.tp.epochs = 16
config.tp.log_train_every = 1000
config.tp.loss = nn.NLLLoss()
config.tp.test_loss = nn.NLLLoss() 
config.tp.use_gpu = False
config.tp.device = torch.device('cuda') if config.tp.use_gpu else torch.device('cpu')

config.opt = Adam
config.op.lr = 1e-3

config.dataset = GMMData
config.dp.device = config.tp.device
config.dp.seed = config.seed
config.dp.gauss_dim = 5
config.dp.num_classes = 5
config.dp.batch_size = 128
config.dp.num_samples = 100000

config.teacher.model = mlp
config.teacher.device = config.tp.device
config.teacher.input_size = config.dp.gauss_dim
config.teacher.hidden_sizes = [8, 32, 16]
config.teacher.output_size = config.dp.num_classes
config.teacher.activation = nn.ReLU()
config.teacher.output_activation= nn.LogSoftmax(dim=1)

config.student.model = mlp
config.student.device = config.tp.device
config.student.input_size = config.dp.gauss_dim
config.student.hidden_sizes = [8, 8, 8]
config.student.output_size = config.dp.num_classes
config.student.activation = nn.ReLU()
config.student.output_activation= nn.LogSoftmax(dim=1)
