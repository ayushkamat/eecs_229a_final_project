from trainers.trainer import Trainer
from torch_utils import *

from collections import defaultdict 
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os
import pickle
import numpy as np
torch.autograd.set_detect_anomaly(True)

class ModifyMNISTTrainer(Trainer):
    """
    This will train Nt teachers, with Ns students each, and compute student accuracy as a function of kldiv 
    """
    def __init__(self, config):
        super().__init__(config)

        # KL Div here can have numerical issues, better with extra precision
        torch.set_default_dtype(torch.float64)
        self.save_path_1 = os.path.join(self.c.weights_path, 'direct_compare.pkl')
        self.save_path_2 = os.path.join(self.c.weights_path, 'student_noise.pkl')

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
        result = {'student_acc':0}
        dataloader = DataLoader(teacher_data, batch_size=self.c.dp.batch_size)
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                x,y = batch
                pred = student(x)
                loss = self.c.tp.test_loss(pred, y)
                result['student_acc'] += (torch.argmax(pred, dim=1) == y).double().mean()
        
        for k in result:
            result[k] = result[k]/(i+1)

        result['teacher_conditional_entropy'] = empirical_posterior_entropy(student_data, teacher, nsamples=10**4).cpu().detach().item()
        result['student_acc'] = result['student_acc'].cpu().detach().item()
        return result

    def run(self):
        self.run1()
        self.run2()

    def run1(self):
        res = []
        Nt = 5
        Ns = 5
        pbar1 = tqdm(total=Nt, position=2, leave=True)
        pbar2 = tqdm(total=Ns*Nt, position=3, leave=True)
        pbar1.set_description("Teacher first run...")
        pbar2.set_description("Student not run...")
        
        teacher_loss = torch.tensor(0)
        student_loss = torch.tensor(0)
        for noise_rate in [0, 0.25, 0.5, 0.75, 1.]:
            for corrupt_rate in [0, 0.25, 0.5, 0.75, 1.]:
                classes = list(range(10))

                for nt in range(Nt):
                    data = self.c.dataset(self.c.dp, classes=classes, noise_rate=noise_rate, corrupt_rate=corrupt_rate)
                    dataloader = DataLoader(data, batch_size=self.c.dp.batch_size)
                    teacher = self.c.teacher.model(self.c.teacher)
                    teacher_opt = self.c.opt(teacher.parameters(), lr = self.c.op.lr)
                    pbar3 = tqdm(total=self.c.tp.epochs*len(data), position=4, leave=False)
                    pbar3.set_description("Teacher {} acc {:.2e} | Noise {} | Corrupt {} | Epoch -1".format(nt, 0, noise_rate, corrupt_rate))

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
                        pbar3.set_description("Teacher {} acc {:.2e} | Noise {} | Corrupt {} | Epoch {}".format(nt, teacher_acc, noise_rate, corrupt_rate, epoch))

                    teacher_result = self.test(teacher, dataloader)
                    teacher_acc = teacher_result['test/acc']
                    teacher_cond_entr = empirical_posterior_entropy(data, teacher, nsamples=10**4).cpu().detach().item()
                    pbar1.set_description("Teacher {} acc {:.2e} | Noise {} | Corrupt {}".format(nt, teacher_acc, noise_rate, corrupt_rate))

                    for ns in range(Ns):
                        student = self.c.student.model(self.c.student)
                        student_opt = self.c.opt(student.parameters(), lr = self.c.op.lr)
                        pbar3 = tqdm(total=self.c.tp.epochs*len(data), position=4, leave=False)
                        pbar3.update(1)

                        for epoch in range(self.c.tp.epochs):
                            for ind, batch in enumerate(dataloader):
                                student_opt.zero_grad()
                                xs = batch[0]
                                ys = teacher(xs)
                                ys = torch.argmax(ys, dim=1)
                                result = self.train(student, (xs, ys), self.c.tp.teacher_loss)
                                student_loss = result['train/loss']
                                student_loss.backward()
                                student_opt.step()
                                pbar3.update(self.c.dp.batch_size)

                        comp = self.compare(teacher, student, data, data)
                        comp['teacher_acc'] = teacher_acc.cpu().detach().item()
                        comp['teacher'] = nt
                        comp['student'] = ns
                        comp['label'] = 'corrupt data p {} and labels p {}'.format(noise_rate, corrupt_rate)
                        comp['true_teacher_cond_entropy'] = teacher_cond_entr
                        student_acc = comp['student_acc']
                        self.iteration += 1
                        pbar2.set_description("Teacher {} acc {:.2e} | Student {} acc {:.2e} | Teacher noise {}".format(nt, teacher_acc, ns, student_acc, noise_rate))
                        pbar2.update(1)
                        res.append(comp)
                        student.cpu()

                        with open(self.save_path_1, 'wb+') as f:
                            pickle.dump(res, f)

                    teacher.cpu()
                    pbar1.update(1)

    def run2(self):
        res = []
        Nt = 5
        Ns = 5
        pbar1 = tqdm(total=Nt, position=2, leave=True)
        pbar2 = tqdm(total=Ns*Nt, position=3, leave=True)
        pbar1.set_description("Teacher first run...")
        pbar2.set_description("Student not run...")
        
        teacher_loss = torch.tensor(0)
        student_loss = torch.tensor(0)
        for nt in range(Nt):
            classes = np.random.permutation(list(range(10)))
            noise_rate, corrupt_rate = 0, 0
            data = self.c.dataset(self.c.dp, classes=classes[:5], noise_rate=noise_rate, corrupt_rate=corrupt_rate)
            dataloader = DataLoader(data, batch_size=self.c.dp.batch_size)
            teacher = self.c.teacher.model(self.c.teacher)
            teacher_opt = self.c.opt(teacher.parameters(), lr = self.c.op.lr)
            pbar3 = tqdm(total=self.c.tp.epochs*len(data), position=4, leave=False)
            pbar3.set_description("Teacher {} acc {:.2e} | Noise {} | Corrupt {}".format(nt, 0, noise_rate, corrupt_rate))

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
                pbar3.set_description("Teacher {} acc {:.2e} | Noise {} | Corrupt {} | Epoch {}".format(nt, teacher_acc, noise_rate, corrupt_rate, epoch))

            teacher_result = self.test(teacher, dataloader)
            teacher_acc = teacher_result['test/acc']
            teacher_cond_entr = empirical_posterior_entropy(data, teacher, nsamples=10**4).cpu().detach().item()
            pbar1.set_description("Teacher {} acc {:.2e} | Noise {} | Corrupt {}".format(nt, teacher_acc, noise_rate, corrupt_rate))

            for ns in range(Ns):
                for noise_rate in [0.25, 0.5, 0.75, 1.]:
                    corrupt_rate = 0
                    student_data = self.c.dataset(self.c.dp, classes=classes[:5], noise_rate=noise_rate, corrupt_rate=corrupt_rate)
                    student_dataloader = DataLoader(student_data, batch_size=self.c.dp.batch_size)
                    student = self.c.student.model(self.c.student)
                    student_opt = self.c.opt(student.parameters(), lr = self.c.op.lr)
                    pbar3 = tqdm(total=self.c.tp.epochs*len(data), position=4, leave=False)
                    pbar3.update(1)

                    for epoch in range(self.c.tp.epochs):
                        for ind, batch in enumerate(dataloader):
                            student_opt.zero_grad()
                            xs = batch[0]
                            ys = teacher(xs)
                            ys = torch.argmax(ys, dim=1)
                            result = self.train(student, (xs, ys), self.c.tp.teacher_loss)
                            student_loss = result['train/loss']
                            student_loss.backward()
                            student_opt.step()
                            pbar3.update(self.c.dp.batch_size)

                    comp = self.compare(teacher, student, data, student_data)
                    comp['teacher_acc'] = teacher_acc.cpu().detach().item()
                    comp['teacher'] = nt
                    comp['student'] = ns
                    comp['label'] = 'data noise p {}'.format(noise_rate)
                    comp['true_teacher_cond_entropy'] = teacher_cond_entr
                    student_acc = comp['student_acc']
                    self.iteration += 1
                    pbar2.set_description("Teacher {} acc {:.2e} | Student {} acc {:.2e} | Student noise {}".format(nt, teacher_acc, ns, student_acc, noise_rate))
                    pbar2.update(1)
                    res.append(comp)
                    student.cpu()

                    with open(self.save_path_2, 'wb+') as f:
                        pickle.dump(res, f)

                for overlap in range(6):
                    corrupt_rate = 0
                    student_data = self.c.dataset(self.c.dp, classes=classes[overlap:overlap+5], noise_rate=noise_rate, corrupt_rate=corrupt_rate)
                    student_dataloader = DataLoader(student_data, batch_size=self.c.dp.batch_size)
                    student = self.c.student.model(self.c.student)
                    student_opt = self.c.opt(student.parameters(), lr = self.c.op.lr)
                    pbar3 = tqdm(total=self.c.tp.epochs*len(data), position=4, leave=False)
                    pbar3.update(1)

                    for epoch in range(self.c.tp.epochs):
                        for ind, batch in enumerate(student_dataloader):
                            student_opt.zero_grad()
                            xs = batch[0]
                            ys = teacher(xs)
                            ys = torch.argmax(ys, dim=1)
                            result = self.train(student, (xs, ys), self.c.tp.teacher_loss)
                            student_loss = result['train/loss']
                            student_loss.backward()
                            student_opt.step()
                            pbar3.update(self.c.dp.batch_size)

                    comp = self.compare(teacher, student, data, student_data)
                    comp['teacher_acc'] = teacher_acc.cpu().detach().item()
                    comp['teacher'] = nt
                    comp['student'] = ns
                    comp['label'] = 'class overlap {}'.format((5.-overlap)/5.)
                    comp['true_teacher_cond_entropy'] = teacher_cond_entr
                    student_acc = comp['student_acc']
                    self.iteration += 1
                    pbar2.set_description("Teacher {} acc {:.2e} | Student {} acc {:.2e} | Overlap {}".format(nt, teacher_acc, ns, student_acc, (5.-overlap)/5.))
                    pbar2.update(1)
                    res.append(comp)
                    student.cpu()

                    with open(self.save_path_2, 'wb+') as f:
                        pickle.dump(res, f)

            teacher.cpu()
            pbar1.update(1)


