from trainers.trainer import Trainer
from collections import defaultdict 
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os
torch.autograd.set_detect_anomaly(True)
from matplotlib import pyplot as plt
import torch.distributions as D

class AdversarialTrainer(Trainer):
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
        if self.c.tp.prior_only:
            generated_input = self.test_dataset.input_prior.rsample((self.c.dp.batch_size,))
        else:
            gaussian_means, gaussian_stds = batch
            generated_distribution = D.MultivariateNormal(gaussian_means, gaussian_stds)
            generated_input = generated_distribution.rsample((self.c.dp.batch_size,))

        teacher_output = self.teacher(generated_input)
        # self.plot_generated_input(generated_input)
        student_output = self.student(generated_input)

        if self.c.tp.prior_only:
            negative_disagreement = 0
            kl_constraint = 0
            generator_loss = 0
        else:
            negative_disagreement = 0.5 * (self.c.tp.generator_loss(teacher_output, student_output) + self.c.tp.generator_loss(student_output, teacher_output))
            kl_constraint = torch.distributions.kl_divergence(generated_distribution, self.test_dataset.input_prior).mean()
            generator_loss = self.c.tp.negative_disagreement_weight * negative_disagreement + self.c.tp.kl_constraint_weight * kl_constraint

        student_loss = self.c.tp.student_loss(student_output, teacher_output.detach())

        result = {'train/student_loss': student_loss,
                  'train/generator_loss': generator_loss,
                  'train/negative_disagreement_loss': negative_disagreement,
                  'train/kl_constraint_loss': kl_constraint,
                  }

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

    def plot_generated_input(self, generated_input):
        """ Plots the first two dimensions of the generated input as well as the means of the test data gaussians """

        for g in self.test_dataset.gaussians:
            plt.scatter(g.mean[0], g.mean[1])
        plt.scatter(generated_input[:, 0].detach().cpu(), generated_input[:, 1].detach().cpu(), color='black')
        plt.savefig(os.path.join(self.c.plots_path, 'generated_iteration_{}'.format(self.iteration)))
        plt.clf()

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
                if ind % self.c.tp.train_student_every != 0 and not self.c.tp.prior_only:
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