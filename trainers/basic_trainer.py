from trainers.trainer import Trainer
from collections import defaultdict 
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os

class BasicTrainer(Trainer):
    """
    Model is a network we are trying to train, and this trainer runs a standard train loop.
    This setup assumes the model is predicting categorical when computing accuracy.
    """
    def __init__(self, config):
        super().__init__(config)
        self.model = self.c.model(self.c.mp)
        self.dataset = self.c.dataset(self.c.dp)
        self.dataloader = DataLoader(self.dataset, batch_size=self.c.dp.batch_size)
        self.test_dataset = self.c.test_dataset(self.c.tdp, mode='test')
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.c.tdp.batch_size)
        self.opt = self.c.opt(self.model.parameters(), lr = self.c.op.lr)

    def train(self, batch):
        result = {}
        x, y = batch
        pred = self.model(x)
        loss = self.c.tp.loss(pred, y)
        result['train/loss'] = loss
        return result

    def test(self):
        result = defaultdict(lambda: 0)
        with torch.no_grad():
            for i, batch in enumerate(self.test_dataloader):
                x,y = batch
                pred = self.model(x)
                loss = self.c.tp.test_loss(pred, y)
                result['test/loss'] += loss
                result['test/acc'] += (torch.argmax(pred, dim=1) == y).float().mean()

        for k in result:
            result[k] = result[k]/(i+1)

        self.log(result)
        return result

    def run(self):
        self.total_iterations = len(self.dataloader)*self.c.tp.epochs
        pbar = tqdm(total=self.total_iterations)
        
        self.test()
        for epoch in range(self.c.tp.epochs):
            for batch in self.dataloader:
                self.opt.zero_grad()
                result = self.train(batch)
                loss = result['train/loss']
                loss.backward()
                self.opt.step()
                if self.iteration % self.c.tp.log_train_every == 0:
                    self.log(result)
                self.iteration += 1
                pbar.set_description("Epoch {}/{} | Loss {:.2e}".format(epoch+1, self.c.tp.epochs, loss.item()))
                pbar.update(1)
            self.test()

        self.save()
    
    def save(self):
        torch.save(self.model.state_dict(), os.path.join(self.c.weights_path, '{}.pth'.format('model')))