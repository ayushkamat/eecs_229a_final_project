from dotmap import DotMap
from torch import nn
from torch.optim import Adam
import torch

from pytorch_utils import empirical_kl
from trainers.distance_trainer import DistanceTrainer
from data.gmm_data import GMMData, GMMTeacherData
from models.mlp import mlp

config = DotMap()
config.seed = 1234

config.trainer = DistanceTrainer
config.tp.epochs = 16
config.tp.log_train_every = 1000
config.tp.loss = nn.NLLLoss()
config.tp.teacher_loss = nn.NLLLoss()
config.tp.student_loss = nn.NLLLoss()
config.tp.test_loss = nn.NLLLoss() 
config.tp.use_gpu = False
config.tp.device = torch.device('cuda') if config.tp.use_gpu else torch.device('cpu')

config.opt = Adam
config.op.lr = 1e-3

config.N_teachers = 50
config.N_students = 10
config.dist_f = empirical_kl
config.dist_scale = 2.5
config.dist_type = 'kldiv'

config.dataset = GMMData
config.dp.device = config.tp.device
config.dp.seed = config.seed
config.dp.gauss_dim = 4
config.dp.num_classes = 4
config.dp.batch_size = 128
config.dp.num_samples = 50000
config.dp.prior = None
config.dp.loc_lower = -2
config.dp.loc_upper = 2

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
config.student.hidden_sizes = [8, 32, 16]
config.student.output_size = config.dp.num_classes
config.student.activation = nn.ReLU()
config.student.output_activation= nn.LogSoftmax(dim=1)
