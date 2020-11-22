from trainers.trainer import Trainer

from collections import defaultdict 
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os
import pickle
torch.autograd.set_detect_anomaly(True)

class CondEntropyTrainer(Trainer):
    """
    This will train Nt teachers then evaluate selecting the dataset by conditional entropy
    """
    def __init__(self, config):
        super().__init__(config)

        # KL Div here can have numerical issues, better with extra precision
        torch.set_default_dtype(torch.float64)
        self.Nt = self.c.N_teachers
        self.dist_scale = self.c.dist_scale
        self.dist_f = self.c.dist_f
        self.dist_type = self.c.dist_type
        self.save_path = os.path.join(self.c.weights_path, 'distance.pkl')

    def train(self, model, batch, loss):
        result = {}
        xs, ys = batch
        pred = model(xs)
        result['train/loss'] = loss(pred, ys)
        return result

    def test(self, model, dataloader):
        result = {'test/loss':0, 'test/acc':0}
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                x,y = batch
                pred = model(x)
                loss = self.c.tp.test_loss(pred, y)
                result['test/loss'] += loss
                result['test/acc'] += (torch.argmax(pred, dim=1) == y).double().mean()

        for k in result:
            result[k] = result[k]/(i+1)
        return result

    def compare(self, teacher, teacher_data, deceive_data):
        result = {'correct_id': 1}
        true_ent = empirical_posterior_entropy(teacher_data, teacher)
        distr = teacher_data
        for data in deceive_data:
            next_ent = empirical_posterior_entropy(data, teacher) for data in deceive_data
            if next_ent < true_ent:
                result['correct_id'] = 0
                distr = data
                break

        result['teacher/loc'] = teacher_data.means
        result['kldiv'] = empirical_kl(teacher_data, distr).cpu().detach().item() if distr is not teacher_data else 0
        return result

    def run(self):
        res = []
        pbar1 = tqdm(total=self.Nt, position=0, leave=True)
        pbar1.set_description("Teacher first run...")
        
        teacher_loss = torch.tensor(0)
        for n in range(self.Nt):
            dataset = self.c.dataset(self.c.dp)
            dataloader = DataLoader(dataset, batch_size=self.c.dp.batch_size)
            teacher = self.c.teacher.model(self.c.teacher)
            teacher_opt = self.c.opt(teacher.parameters(), lr = self.c.op.lr)
            pbar2 = tqdm(total=self.c.tp.epochs*self.c.dp.num_samples, position=2, leave=False)
            for epoch in range(self.c.tp.epochs):
                for ind, batch in enumerate(dataloader):
                    teacher_opt.zero_grad()
                    result = self.train(teacher, batch, self.c.tp.teacher_loss)
                    teacher_loss = result['train/loss']
                    teacher_loss.backward()
                    teacher_opt.step()
                    pbar2.update(self.c.dp.batch_size)
            teacher_result = self.test(teacher, dataloader)
            teacher_acc = teacher_result['test/acc']
            pbar1.set_description("Teacher {} acc {:.2e}".format(n, teacher_acc))

            for std in self.c.std:
                deceive_data = []
                for m in range(self.Nd):
                    data = dataset.copy(std=std)
                    deceive_data.append(data)

                comp = self.compare(teacher, dataset, deceive_data)
                comp['test/teacher_acc'] = teacher_acc.cpu().detach().item()
                comp['teacher'] = n
                comp['std'] = std
                self.iteration += 1
                pbar1.set_description("Teacher {} acc {:.2e} | correct id: {}".format(n, teacher_acc, comp['correct_id']))
                res.append(comp)

                with open(self.save_path, 'wb+') as f:
                    pickle.dump(res, f)

            teacher.cpu()
            pbar1.update(1)
