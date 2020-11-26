from dotmap import DotMap
from trainers.dafl_trainer import DAFLTrainer
from data.adversarial_toy_gauss_data import AdversarialToyGauss
from data.toy_gauss_data import ToyGauss
from models.mlp import mlp
from models.generator import generator
from torch import nn
from torch.optim import Adam
import torch

config = DotMap()
config.seed = 1

config.trainer = DAFLTrainer
config.tp.epochs = 16
config.tp.log_train_every = 50
config.tp.train_gen_every = 2
config.tp.test_loss = nn.NLLLoss() 
config.tp.student_loss = nn.KLDivLoss(log_target=True, reduction='batchmean')
config.tp.use_gpu = False
config.tp.device = torch.device('cuda') if config.tp.use_gpu else torch.device('cpu')

config.opt = Adam
config.op.lr = 1e-3

config.dataset = AdversarialToyGauss
config.dp.device = config.tp.device
config.dp.seed = config.seed
config.dp.gauss_dim = 3
config.dp.num_classes = 3
config.dp.batch_size = 512
config.dp.num_samples = 100000

config.test_dataset = ToyGauss
config.tdp.device = config.tp.device
config.tdp.seed = config.seed
config.tdp.gauss_dim = 3
config.tdp.num_classes = 3
config.tdp.batch_size = 128
config.tdp.num_samples = 1000

config.generator.model = generator
config.generator.device = config.tp.device
config.generator.noise_dim = 100
config.generator.noise_mean = 0
config.generator.noise_std = 1
config.generator.input_size = config.generator.noise_dim
config.generator.hidden_sizes = [16, 16, 16]
config.generator.output_size = config.dp.gauss_dim
config.generator.activation = nn.ReLU()
config.generator.output_activation= nn.Identity()

config.teacher.model = mlp
config.teacher.device = config.tp.device
config.teacher.checkpoint = 'logs/toy_config@1606134102/weights/model.pth'
config.teacher.input_size = config.dp.gauss_dim
config.teacher.hidden_sizes = [8, 32, 16]
config.teacher.output_size = config.dp.num_classes
config.teacher.activation = nn.ReLU()
config.teacher.output_activation= nn.LogSoftmax(dim=1)

config.student.model = mlp
config.student.device = config.tp.device
config.student.input_size = config.dp.gauss_dim
config.student.hidden_sizes = [4, 12, 8]
config.student.output_size = config.dp.num_classes
config.student.activation = nn.ReLU()
config.student.output_activation= nn.LogSoftmax(dim=1)
