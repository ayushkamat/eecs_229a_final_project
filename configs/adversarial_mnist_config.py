from dotmap import DotMap
from trainers.adversarial_trainer import AdversarialTrainer
from data.mnist_data import MNISTData
from data.adversarial_toy_gauss_data import AdversarialToyGauss
from models.mlp import mlp
from models.generator import generator
from torch import nn
from torch.optim import Adam
import torch

config = DotMap()
config.seed = 1

config.trainer = AdversarialTrainer
config.tp.epochs = 16
config.tp.log_train_every = 50
config.tp.train_gen_every = 2
config.tp.generator_loss = lambda input, target : -nn.KLDivLoss(log_target=True, reduction='batchmean')(input, target)
config.tp.student_loss = nn.KLDivLoss(log_target=True, reduction='batchmean')
config.tp.test_loss = nn.NLLLoss() 
config.tp.use_gpu = False
config.tp.device = torch.device('cuda') if config.tp.use_gpu else torch.device('cpu')

config.opt = Adam
config.op.lr = 1e-3

config.dataset = AdversarialToyGauss
config.dp.device = config.tp.device
config.dp.seed = config.seed
config.dp.gauss_dim = 28 * 28
config.dp.num_classes = 6
config.dp.batch_size = 128
config.dp.num_samples = 100000

config.test_dataset = MNISTData
config.dp.device = config.tp.device
config.tdp.seed = 1
config.tdp.classes = [0, 1, 2, 3, 4, 5]
config.tdp.resolution = (28, 28)
config.tdp.dir ='./data/cache/mnist/'
config.tdp.num_classes = config.dp.num_classes
config.tdp.batch_size = 128

config.generator.model = generator
config.generator.device = config.tp.device
config.generator.noise_dim = 100
config.generator.noise_mean = 0
config.generator.noise_std = 1
config.generator.input_size = config.generator.noise_dim
config.generator.hidden_sizes = [8, 32, 16]
config.generator.output_size = config.dp.gauss_dim
config.generator.activation = nn.ReLU()
config.generator.output_activation= nn.Identity()

config.teacher.model = mlp
config.teacher.device = config.tp.device
config.teacher.checkpoint = 'logs/mnist_basic_config@1605591728/weights/model.pth'
config.teacher.input_size = 28 * 28
config.teacher.hidden_sizes = [256, 256, 256]
config.teacher.output_size = config.dp.num_classes
config.teacher.activation = nn.ReLU()
config.teacher.output_activation= nn.LogSoftmax(dim=1)

config.student.model = mlp
config.student.device = config.tp.device
config.student.input_size = 28 * 28
config.student.hidden_sizes = [128, 128, 128]
config.student.output_size = config.dp.num_classes
config.student.activation = nn.ReLU()
config.student.output_activation= nn.LogSoftmax(dim=1)
