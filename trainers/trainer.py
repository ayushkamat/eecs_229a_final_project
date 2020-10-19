from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, config):
        self.c = config
        self.writer = SummaryWriter(self.c.exp_path)
        self.iteration = 0

    def log(self, result):
        for name, value in result.items():
            self.writer.add_scalar(name, float(value), self.iteration)
        del result

    def run():
        raise NotImplementedError

    def save():
        raise NotImplementedError