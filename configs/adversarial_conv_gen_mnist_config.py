from dotmap import DotMap
from trainers.adversarial_conv_gen_trainer import AdversarialConvTrainer
from data.adversarial_mnist_data import AdversarialMNISTData
from data.mnist_data import MNISTData
from models.mlp import mlp
from models.conv_generator import ConvGenerator

from torch import nn
from torch.optim import Adam
import torch

config = DotMap()
config.seed = 1

config.trainer = AdversarialConvTrainer
config.tp.epochs = 64
config.tp.log_train_every = 50
config.tp.train_student_every = 25
config.tp.generator_loss = lambda input, target : -nn.KLDivLoss(log_target=True, reduction='batchmean')(input, target)
config.tp.student_loss = nn.KLDivLoss(log_target=True, reduction='batchmean')
config.tp.test_loss = nn.NLLLoss() 
config.tp.use_gpu = True
config.tp.device = torch.device('cuda') if config.tp.use_gpu else torch.device('cpu')
config.tp.negative_disagreement_weight = 1
config.tp.kl_constraint_weight = 1

config.opt = Adam
config.op.lr = 1e-3

config.dataset = MNISTData
config.dp.device = config.tp.device
config.dp.seed = config.seed # seed must match between test and train in order to use same underlying gaussians
config.dp.classes = [0, 1, 2, 3, 4, 5]
config.dp.resolution = (28, 28)
config.dp.dir = './data/cache/mnist/'
config.dp.num_classes = 6
config.dp.batch_size = 32

config.test_dataset = MNISTData
config.tdp.device = config.tp.device
config.tdp.seed = config.seed # seed must match between test and train in order to use same underlying gaussians
config.tdp.classes = [0, 1, 2, 3, 4, 5]
config.tdp.resolution = (28, 28)
config.tdp.dir = './data/cache/mnist/'
config.tdp.num_classes = 6
config.tdp.batch_size = 128

config.generator.model = ConvGenerator
config.generator.device = config.tp.device
config.generator.num_samples = 100
config.generator.input_size = 100
config.generator.hidden_sizes = [512, 512, 512]
config.generator.resolution = 28
config.generator.output_size = 28 * 28
config.generator.activation = nn.ReLU()
config.generator.output_activation= nn.Identity()

config.teacher.model = mlp
config.teacher.device = config.tp.device
config.teacher.checkpoint = 'logs/mnist_basic_config@1606815386/weights/model.pth'
config.teacher.input_size = config.tdp.resolution[0] * config.tdp.resolution[1]
config.teacher.hidden_sizes = [256, 256, 256]
config.teacher.output_size = config.dp.num_classes
config.teacher.activation = nn.ReLU()
config.teacher.output_activation= nn.LogSoftmax(dim=1)

config.student.model = mlp
config.student.device = config.tp.device
config.student.input_size = config.tdp.resolution[0] * config.tdp.resolution[1]
config.student.hidden_sizes = [256, 256, 256]
config.student.output_size = config.dp.num_classes
config.student.activation = nn.ReLU()
config.student.output_activation= nn.LogSoftmax(dim=1)
