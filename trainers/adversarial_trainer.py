from trainers.trainer import Trainer
from collections import defaultdict 
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os

class AdversarialTrainer(Trainer):
    """
    Teacher is a trained model whose weights are loaded from a checkpoint.
    Student is a new model, potentially of a different architecture.
    The student will learn by matching its prediction to what the teacher predicts for a given input.
    In the case of TeacherToyGauss, the input is randomly sampled using torch.rand
    """
    def __init__(self, config):
        super().__init__(config)
        self.teacher = self.c.teacher.model(self.c.teacher)
        self.student = self.c.student.model(self.c.student)
        self.teacher.load_state_dict(torch.load(self.c.teacher.checkpoint))
        self.c.dp.teacher = self.teacher
        self.dataset = self.c.dataset(self.c.dp)
        self.dataloader = DataLoader(self.dataset, batch_size=self.c.dp.batch_size)
        self.test_dataset = self.c.test_dataset(self.c.tdp, mode='test')
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.c.tdp.batch_size)
        self.stud_opt = self.c.opt(self.student.parameters(), lr = self.c.op.lr)
        self.gen_opt = self.c.opt(self.generator.parameters(), lr = self.c.op.lr)

    def train(self, batch_size):
        # x sampled from some latent distribution
        out = self.generator(batch_size)
        y1 = self.teacher(out)
        y2 = self.student(out)
        # generator loss = -kldivloss - entropy loss
        
        # possibly condition the generator on a label 
        # + add cross-entropy loss between input label and teacher's output
        # --> In this scenario, we don't need adversarial training
        generator_loss = self.c.tp.generator_loss(y1, y2)

        # student_loss = kldivloss
        student_loss = self.c.tp.student_loss(y1, y2)
        result = {'student_loss': student_loss,
                  'generator_loss': generator_loss}
        return result

    def test(self):
        result = defaultdict(lambda: 0)
        with torch.no_grad():
            for i, batch in enumerate(self.test_dataloader):
                x,y = batch
                pred = self.student(x)
                loss = self.c.tp.test_loss(pred, y)
                result['loss_test'] += loss
                result['acc_test'] += (torch.argmax(pred, dim=1) == y).float().mean()

        for k in result:
            result[k] = result[k]/(i+1)

        self.log(result)
        return result

    def run(self):
        self.total_iterations = len(self.dataloader)*self.c.tp.epochs
        pbar = tqdm(total=self.total_iterations)
        
        self.test()
        for epoch in range(self.c.tp.epochs):
            for _ in range(iterations):
                self.stud_opt.zero_grad()
                self.gen_opt.zero_grad()
                result = self.train(batch_size)
                gen_loss = result['generator_loss']
                stud_loss = result['student_loss']
                for g in range(self.c.tp.gen_iters):
                    gen_loss.backward()
                    self.gen_opt.step()

                for d in range(self.c.tp.stud_iters):
                    stud_loss.backward()
                    self.stud_opt.step()
                if self.iteration % self.c.tp.log_train_every == 0:
                    self.log(result)
                self.iteration += 1
                pbar.set_description("Epoch {}/{} | Loss {}".format(epoch+1, self.c.tp.epochs, loss.item()))
                pbar.update(1)
            self.test()

        self.save()
    
    def save(self):
        torch.save(self.student.state_dict(), os.path.join(self.c.weights_path, '{}.pth'.format('student')))
        torch.save(self.teacher.state_dict(), os.path.join(self.c.weights_path, '{}.pth'.format('teacher')))