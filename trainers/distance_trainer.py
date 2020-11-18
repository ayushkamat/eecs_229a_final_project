from trainers.trainer import Trainer
from pytorch_utils import empirical_kl

from collections import defaultdict 
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os
import pickle
torch.autograd.set_detect_anomaly(True)

class DistanceTrainer(Trainer):
    """
    This will train Nt teachers, with Ns students each, and compute student accuracy as a function of kldiv 
    """
    def __init__(self, config):
        super().__init__(config)
        self.Nt = self.c.N_teachers
        self.Ns = self.c.N_students
        self.save_path = os.path.join(self.c.weights_path, 'distance.pkl')

    def train(self, model, batch, loss):
        result = {}
        xs, ys = batch
        pred = model(xs)
        result['train/loss'] = loss(pred, ys)
        return result

    def test(self, model, dataloader):
        result = defaultdict(lambda: 0)
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                x,y = batch
                pred = model(x)
                loss = self.c.tp.test_loss(pred, y)
                result['test/loss'] += loss
                result['test/acc'] += (torch.argmax(pred, dim=1) == y).float().mean()

        for k in result:
            result[k] = result[k]/(i+1)
        return result

    def compare(self, teacher, student, teacher_data, student_data):
        result = defaultdict(lambda: 0)
        dataloader = DataLoader(teacher_data, batch_size=self.c.dp.batch_size)
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                x,y = batch
                pred = self.student(x)
                loss = self.c.tp.test_loss(pred, y)
                result['test/acc'] += (torch.argmax(pred, dim=1) == y).float().mean()
        
        for k in result:
            result[k] = result[k]/(i+1)

        result['kldiv'] = empirical_kl(teacher_data, student_data)
        return result

    def run(self):
        res = []
        pbar1 = tqdm(total=self.Ns*self.Nt, position=0)
        pbar2 = tqdm(total=self.Ns*self.Nt, position=2)
        
        teacher_loss = torch.tensor(0)
        student_loss = torch.tensor(0)
        for n in range(self.Nt):
            dataset = self.c.dataset(self.c.dp)
            dataloader = DataLoader(dataset, batch_size=self.c.dp.batch_size)
            teacher = self.c.teacher.model(self.c.teacher)
            teacher_opt = self.c.opt(teacher.parameters(), lr = self.c.op.lr)
            for epoch in range(self.c.tp.epochs):
                for ind, batch in enumerate(dataloader):
                    teacher.zero_grads
                    result = self.train(teacher, batch, self.c.tp.teacher_loss)
                    teacher_loss = result['train/loss']
                    teacher_loss.backward()
                    teacher_opt.step()
            teacher_result = self.test(teacher, dataloader)
            teacher_acc = teacher_result['test/acc']
            pbar1.set_description("Teacher {} acc {:.2e}".format(n, teacher_acc))
            pbar1.update(1)

            for m in range(self.Ns):
                student_data = self.c.dataset(self.c.dp)
                student_loader = DataLoader(student_data, batch_size=self.c.dp.batch_size)
                student = self.c.student.model(self.c.student)
                student_opt = self.c.opt(student.parameters(), lr = self.c.op.lr)
                for epoch in range(self.c.tp.epochs):
                    for ind, batch in enumerate(student_loader):
                        student.zero_grads()
                        xs = batch[0]
                        ys = self.teacher(xs)
                        result = self.train(student, (xs, ys), self.c.tp.teacher_loss)
                        student_loss = result['train/loss']
                        student_loss.backward()
                        student_opt.step()

                comp = self.compare(teacher, student, dataset, student_data)
                comp['test/teacher_acc'] = teacher_acc
                student_acc = comp['test/acc']
                kldiv = comp['kldiv']
                self.iteration += 1
                pbar2.set_description("Teacher {} acc {:.2e} | Student {} acc {:.2e} | kl div {:.2e}".format(n, teacher_acc, m, student_acc, kldiv))
                pbar2.update(1)
                res.append(comp)

                with open(self.save_path, 'wb+') as f:
                    pickle.dump(res, f)
