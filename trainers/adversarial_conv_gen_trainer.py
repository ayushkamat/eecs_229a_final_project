from trainers.trainer import Trainer
from collections import defaultdict 
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from torch.autograd import Variable

import os
torch.autograd.set_detect_anomaly(True)

class AdversarialConvTrainer(Trainer):
    """
    Teacher is a trained model whose weights are loaded from a checkpoint.
    Student is a new model, potentially of a different architecture.
    The student will learn by matching its prediction to what the teacher predicts for a generated input
    The generator will have a loss of its own, which will hopefully encourage more structure to the generated inputs
    """
    def __init__(self, config):
        super().__init__(config)
        self.teacher = self.c.teacher.model(self.c.teacher)
        self.student = self.c.student.model(self.c.student)
        self.generator = self.c.generator.model(self.c.generator)
        self.teacher.load_state_dict(torch.load(self.c.teacher.checkpoint))
        self.c.dp.teacher = self.teacher
        self.c.dp.generator = self.generator
        self.dataset = self.c.dataset(self.c.dp)
        self.dataloader = DataLoader(self.dataset, batch_size=self.c.dp.batch_size)
        self.test_dataset = self.c.test_dataset(self.c.tdp, mode='test')
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.c.tdp.batch_size)
        self.stud_opt = self.c.opt(self.student.parameters(), lr = self.c.op.lr)
        self.gen_opt = self.c.opt(self.generator.parameters(), lr = self.c.op.lr)

    def train(self, batch):
        data, labels = batch
        z = Variable(torch.randn(self.c.dp.batch_size, self.c.generator.num_samples)).cuda()
        generated_inputs = torch.flatten(torch.squeeze(self.generator(z)), 1, 2)
        teacher_output = self.teacher(generated_inputs)
        student_output = self.student(generated_inputs)

        generator_loss = 0.5 * (self.c.tp.generator_loss(teacher_output, student_output) + self.c.tp.generator_loss(student_output, teacher_output))
        student_loss = self.c.tp.student_loss(student_output, teacher_output.detach())

        result = {'train/student_loss': student_loss,
                  'train/generator_loss': generator_loss}

        return result

    def test(self):
        result = defaultdict(lambda: 0)
        with torch.no_grad():
            for i, batch in enumerate(self.test_dataloader):
                x,y = batch
                pred = self.student(x)
                loss = self.c.tp.test_loss(pred, y)
                result['test/loss'] += loss
                result['test/acc'] += (torch.argmax(pred, dim=1) == y).float().mean()

        for k in result:
            result[k] = result[k]/(i+1)

        self.log(result)
        return result

    def zero_grads(self):
        self.teacher.zero_grad()
        self.generator.zero_grad()
        self.student.zero_grad()

    def run(self):
        self.total_iterations = len(self.dataloader)*self.c.tp.epochs
        pbar = tqdm(total=self.total_iterations)
        
        self.test()
        generator_loss = torch.tensor(0)
        student_loss = torch.tensor(0)
        for epoch in range(self.c.tp.epochs):
            for ind, batch in enumerate(self.dataloader):
                self.zero_grads()
                if ind % self.c.tp.train_student_every != 0:
                    result = self.train(batch)
                    generator_loss = result['train/generator_loss']
                    generator_loss.backward()
                    self.gen_opt.step()
                else:
                    result = self.train(batch)
                    student_loss = result['train/student_loss']
                    student_loss.backward()
                    self.stud_opt.step()

                if self.iteration % self.c.tp.log_train_every == 0:
                    self.log(result)
                self.iteration += 1
                pbar.set_description("Epoch {}/{} | Gen Loss {:.2e} Student Loss {:.2e}".format(epoch+1, self.c.tp.epochs, generator_loss.item(), student_loss.item()))
                pbar.update(1)
            self.test()

        self.save()
    
    def save(self):
        torch.save(self.generator.state_dict(), os.path.join(self.c.weights_path, '{}.pth'.format('generator')))
        torch.save(self.student.state_dict(), os.path.join(self.c.weights_path, '{}.pth'.format('student')))
        torch.save(self.teacher.state_dict(), os.path.join(self.c.weights_path, '{}.pth'.format('teacher')))