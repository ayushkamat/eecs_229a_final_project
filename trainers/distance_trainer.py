from trainers.trainer import Trainer

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

        # KL Div here can have numerical issues, better with extra precision
        torch.set_default_dtype(torch.float64)
        self.Nt = self.c.N_teachers
        self.Ns = self.c.N_students
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

    def compare(self, teacher, student, teacher_data, student_data):
        result = {'test/student_acc':0}
        dataloader = DataLoader(teacher_data, batch_size=self.c.dp.batch_size)
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                x,y = batch
                pred = student(x)
                loss = self.c.tp.test_loss(pred, y)
                result['test/student_acc'] += (torch.argmax(pred, dim=1) == y).double().mean()
        
        for k in result:
            result[k] = result[k]/(i+1)

        result[self.dist_type] = self.dist_f(teacher_data, student_data)
        result['student/loc'] = student_data.means
        result['teacher/loc'] = teacher_data.means
        return result

    def run(self):
        res = []
        pbar1 = tqdm(total=self.Nt, position=0, leave=True)
        pbar2 = tqdm(total=self.Ns*self.Nt, position=1, leave=True)
        pbar1.set_description("Teacher first run...")
        pbar2.set_description("Student not run...")
        
        teacher_loss = torch.tensor(0)
        student_loss = torch.tensor(0)
        for n in range(self.Nt):
            dataset = self.c.dataset(self.c.dp)
            dataloader = DataLoader(dataset, batch_size=self.c.dp.batch_size)
            teacher = self.c.teacher.model(self.c.teacher)
            teacher_opt = self.c.opt(teacher.parameters(), lr = self.c.op.lr)
            pbar3 = tqdm(total=self.c.tp.epochs*self.c.dp.num_samples, position=2, leave=False)
            for epoch in range(self.c.tp.epochs):
                for ind, batch in enumerate(dataloader):
                    teacher_opt.zero_grad()
                    result = self.train(teacher, batch, self.c.tp.teacher_loss)
                    teacher_loss = result['train/loss']
                    teacher_loss.backward()
                    teacher_opt.step()
                    pbar3.update(self.c.dp.batch_size)
            teacher_result = self.test(teacher, dataloader)
            teacher_acc = teacher_result['test/acc']
            pbar1.set_description("Teacher {} acc {:.2e}".format(n, teacher_acc))

            for m in range(self.Ns):
                std = float(m) / self.Ns
                std *= self.dist_scale
                student_data = dataset.copy(std=std)
                student_loader = DataLoader(student_data, batch_size=self.c.dp.batch_size)
                student = self.c.student.model(self.c.student)
                student_opt = self.c.opt(student.parameters(), lr = self.c.op.lr)
                pbar3 = tqdm(total=self.c.tp.epochs*self.c.dp.num_samples, position=2, leave=False)
                pbar3.update(1)
                for epoch in range(self.c.tp.epochs):
                    for ind, batch in enumerate(student_loader):
                        student_opt.zero_grad()
                        xs = batch[0]
                        ys = teacher(xs)
                        ys = torch.argmax(ys, dim=1)
                        result = self.train(student, (xs, ys), self.c.tp.teacher_loss)
                        student_loss = result['train/loss']
                        student_loss.backward()
                        student_opt.step()
                        pbar3.update(self.c.dp.batch_size)

                comp = self.compare(teacher, student, dataset, student_data)
                comp['test/teacher_acc'] = teacher_acc
                comp['teacher'] = n
                comp['student'] = m
                student_acc = comp['test/student_acc']
                kldiv = comp['kldiv']
                self.iteration += 1
                pbar2.set_description("Teacher {} acc {:.2e} | Student {} acc {:.2e} | kl div {:.2e}".format(n, teacher_acc, m, student_acc, kldiv))
                pbar2.update(1)
                res.append(comp)
                student.cpu()

                with open(self.save_path, 'wb+') as f:
                    pickle.dump(res, f)

            teacher.cpu()
            pbar1.update(1)
