from dotmap import DotMap
from trainers.adversarial_trainer import AdversarialTrainer
from trainers.decoupled_generative_trainer import DecoupledGenerativeTrainer
from trainers.generative_trainer import GenerativeTrainer
from data.adversarial_toy_gauss_data import AdversarialToyGauss
from data.toy_gauss_data import ToyGauss
from data.mnist_data import MNISTData
from models.mlp import mlp
from models.generator import generator
from models.generative_approach_generator import GenerativeGenerator
from torch import nn, distributions
from torch.optim import Adam
import torch

config = DotMap()
config.seed = 1

config.trainer = DecoupledGenerativeTrainer
config.tp.gen_epochs = 4
config.tp.stu_epochs = 48
config.tp.log_train_every = 1
config.tp.generator_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)
config.tp.student_loss = nn.NLLLoss()
config.tp.entropy_loss = lambda mean, sigma: torch.norm(sigma) # torch.norm(distributions.kl.kl_divergence(distributions.normal.Normal(mean, sigma), 
																					                     # distributions.uniform.Uniform(torch.empty(*mean.shape).fill_(config.dp.min_val), 
																										               		           # torch.empty(*mean.shape).fill_(config.dp.max_val))))
config.tp.test_loss = lambda input, target: torch.log(nn.MSELoss()(input, target))
config.tp.use_gpu = False
config.tp.device = torch.device('cuda') if config.tp.use_gpu else torch.device('cpu')

config.opt = Adam
config.op.lr = 1e-2

config.dataset = MNISTData
config.dp.device = config.tp.device
config.dp.seed = config.seed
config.dp.classes = [0, 1, 2, 3, 4, 5]
config.dp.resolution = (28, 28)
config.dp.dir = "./data/cache/mnist"
config.dp.num_classes = 6
config.dp.batch_size = 512
config.dp.num_samples = 10000

# config.dataset = MNISTData
# config.dp.device = config.tp.device
# config.dp.seed = config.seed # seed must match between test and train in order to use same underlying gaussians
# config.dp.classes = [0, 1, 2, 3, 4, 5]
# config.dp.resolution = (28, 28)
# config.dp.dir = './data/cache/mnist/'
# config.dp.num_classes = 10
# config.dp.batch_size = 128

# config.test_dataset = ToyGauss
# config.tdp.device = config.tp.device
# config.tdp.seed = config.seed
# config.tdp.gauss_dim = 100
# config.tdp.num_classes = 2
# config.tdp.batch_size = 128
# config.tdp.num_samples = 1000

config.test_dataset = MNISTData
config.tdp.device = config.tp.device
config.tdp.seed = config.seed
config.tdp.classes = config.dp.classes
config.tdp.resolution = config.dp.resolution
config.tdp.dir = config.dp.dir
config.tdp.num_classes = config.dp.num_classes
config.tdp.batch_size = 256

config.generator.model = GenerativeGenerator
config.generator.device = config.tp.device
config.generator.load_gen = False
config.generator.checkpoint = 'logs/decoupled_generative_mnist_config@1606783641/weights/generator.pth'
config.generator.num_samples = 100
config.generator.input_size = config.dp.num_classes
config.generator.hidden_sizes = [256, 256, 256]
config.generator.output_size = 28 * 28
config.generator.activation = nn.ReLU()
config.generator.output_activation= nn.Identity()

config.teacher.model = mlp
config.teacher.device = config.tp.device
config.teacher.checkpoint = 'logs/mnist_basic_config@1606795480/weights/model.pth'
config.teacher.input_size = 784
config.teacher.hidden_sizes = [256, 256, 256]
config.teacher.output_size = config.dp.num_classes
config.teacher.activation = nn.ReLU()
config.teacher.output_activation= nn.LogSoftmax(dim=1)

# config.model = mlp
# config.mp.device = config.tp.device
# config.mp.input_size = config.dp.resolution[0] * config.dp.resolution[1]
# config.mp.hidden_sizes = [256, 256, 256]
# config.mp.output_size = config.dp.num_classes
# config.mp.activation = nn.ReLU()
# config.mp.output_activation= nn.Identity()

config.student.model = mlp
config.student.device = config.tp.device
config.student.input_size = 784
config.student.hidden_sizes = [256, 256, 256]
config.student.output_size = config.dp.num_classes
config.student.activation = nn.ReLU()
config.student.output_activation= nn.LogSoftmax(dim=1)

# from dotmap import DotMap
# from trainers.basic_trainer import BasicTrainer
# from data.mnist_data import MNISTData
# from models.mlp import mlp
# from torch import nn
# from torch.optim import Adam
# import torch

# config = DotMap()
# config.seed = 1

# config.trainer = BasicTrainer
# config.tp.epochs = 16
# config.tp.log_train_every = 1000
# config.tp.loss = nn.CrossEntropyLoss()
# config.tp.test_loss = nn.CrossEntropyLoss() # train and test separate for flexibility
# config.tp.device = torch.device('cuda') if config.tp.use_gpu else torch.device('cpu')
# config.opt = Adam
# config.op.lr = 1e-3




# config.test_dataset = MNISTData
# config.tdp.device = config.tp.device
# config.tdp.seed = config.seed
# config.tdp.classes = config.dp.classes
# config.tdp.resolution = config.dp.resolution
# config.tdp.dir = config.dp.dir
# config.tdp.num_classes = config.dp.num_classes
# config.tdp.batch_size = 128

