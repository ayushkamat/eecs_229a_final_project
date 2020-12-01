from dotmap import DotMap
from torch import nn
from torch.optim import Adam
import torch

from torch_utils import empirical_kl
from trainers.modify_mnist_trainer import ModifyMNISTTrainer
from data.modify_mnist_data import MNISTData
from models.lenet5 import lenet5

config = DotMap()
config.seed = 1234

config.trainer = ModifyMNISTTrainer
config.tp.epochs = 5
config.tp.log_train_every = 1000
config.tp.loss = nn.NLLLoss()
config.tp.teacher_loss = nn.NLLLoss()
config.tp.student_loss = nn.NLLLoss()
config.tp.test_loss = nn.NLLLoss() 
config.tp.use_gpu = True
config.tp.device = torch.device('cuda') if config.tp.use_gpu else torch.device('cpu')

config.opt = Adam
config.op.lr = 1e-3

config.dataset = MNISTData
config.dp.device = config.tp.device
config.dp.seed = config.seed
config.dp.batch_size = 128
config.dp.resolution = (28, 28)
config.dp.num_classes = 10

config.teacher.model = lenet5
config.teacher.device = config.tp.device
config.teacher.input_size = config.dp.resolution[0] * config.dp.resolution[1]
config.teacher.output_size = config.dp.num_classes
config.teacher.activation = nn.ReLU()
config.teacher.output_activation= nn.LogSoftmax(dim=1)

config.student.model = lenet5
config.student.device = config.tp.device
config.student.input_size = config.dp.resolution[0] * config.dp.resolution[1]
config.student.output_size = config.dp.num_classes
config.student.activation = nn.ReLU()
config.student.output_activation= nn.LogSoftmax(dim=1)
