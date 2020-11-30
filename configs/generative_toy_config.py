from dotmap import DotMap
from trainers.adversarial_trainer import AdversarialTrainer
from trainers.generative_trainer import GenerativeTrainer
from data.adversarial_toy_gauss_data import AdversarialToyGauss
from data.toy_gauss_data import ToyGauss
from models.mlp import mlp
from models.generator import generator
from models.generative_approach_generator import GenerativeGenerator
from torch import nn, distributions
from torch.optim import Adam
import torch

config = DotMap()
config.seed = 1

config.trainer = GenerativeTrainer
config.tp.epochs = 64
config.tp.log_train_every = 10
config.tp.train_gen_every = 16
config.tp.generator_loss = lambda input, target : -nn.KLDivLoss(reduction='batchmean', log_target=True)(input, target)
config.tp.student_loss = nn.NLLLoss()
config.tp.entropy_loss = lambda mean, sigma: torch.norm(sigma)# torch.norm(distributions.kl.kl_divergence(distributions.normal.Normal(mean, sigma), 
																					   # distributions.uniform.Uniform(torch.empty(*mean.shape).fill_(config.dp.min_val), 
																												     # torch.empty(*mean.shape).fill_(config.dp.max_val))))
config.tp.test_loss = nn.MSELoss() 
config.tp.use_gpu = False
config.tp.device = torch.device('cuda') if config.tp.use_gpu else torch.device('cpu')

config.opt = Adam
config.op.lr = 1e-4

config.dataset = ToyGauss
config.dp.min_val = -10
config.dp.max_val = 10
config.dp.device = config.tp.device
config.dp.seed = config.seed
config.dp.gauss_dim = 2
config.dp.num_classes = 3
config.dp.batch_size = 128
config.dp.num_samples = 100

config.test_dataset = ToyGauss
config.tdp.device = config.tp.device
config.tdp.seed = config.seed
config.tdp.gauss_dim = 2
config.tdp.num_classes = 3
config.tdp.batch_size = 128
config.tdp.num_samples = 1000

config.generator.model = GenerativeGenerator
config.generator.device = config.tp.device
config.generator.num_samples = 100
config.generator.input_size = 1
config.generator.hidden_sizes = [8, 32, 16]
config.generator.output_size = config.dp.gauss_dim
config.generator.activation = nn.ReLU()
config.generator.output_activation= nn.Identity()

config.teacher.model = mlp
config.teacher.device = config.tp.device
config.teacher.checkpoint = 'logs/toy_config@1606389164/weights/model.pth'
config.teacher.input_size = config.dp.gauss_dim
config.teacher.hidden_sizes = [8, 32, 16]
config.teacher.output_size = config.dp.num_classes
config.teacher.activation = nn.ReLU()
config.teacher.output_activation= nn.LogSoftmax(dim=1)

config.student.model = mlp
config.student.device = config.tp.device
config.student.input_size = config.dp.gauss_dim
config.student.hidden_sizes = [4, 8, 4]
config.student.output_size = config.dp.num_classes
config.student.activation = nn.ReLU()
config.student.output_activation= nn.LogSoftmax(dim=1)