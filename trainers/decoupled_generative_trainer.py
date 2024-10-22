from trainers.trainer import Trainer
from collections import defaultdict 
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os
torch.autograd.set_detect_anomaly(True)
import torch.distributions as D

class DecoupledGenerativeTrainer(Trainer):
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
        self.load_gen = self.c.generator.load_gen
        if self.c.generator.load_gen:
            self.generator.load_state_dict(torch.load(self.c.generator.checkpoint))
        self.teacher.load_state_dict(torch.load(self.c.teacher.checkpoint))
        self.c.dp.teacher = self.teacher
        self.c.dp.generator = self.generator
        self.dataset = self.c.dataset(self.c.dp)
        self.dataloader = DataLoader(self.dataset, batch_size=self.c.dp.batch_size)
        self.test_dataset = self.c.test_dataset(self.c.tdp, mode='test')
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.c.tdp.batch_size)
        self.stud_opt = self.c.opt(self.student.parameters(), lr = self.c.op.lr)
        self.gen_opt = self.c.opt(self.generator.parameters(), lr = self.c.op.lr)
        self.num_classes = self.c.dp.num_classes
        self.student_loss = 0
        self.student_acc = 0
        self.teacher_acc = 0
        self.true_student_acc = 0
        self.kl_constraint = 0
        self.coherence = 0
        self.generator_loss = 0

    def train(self, batch, train_gen=True):
        data, labels = batch
        labels_oh = torch.nn.functional.one_hot(labels).float()
        means, sigmas = self.generator(labels_oh)
        sigmas = torch.abs(sigmas) # ensure absolute values
        dim = means.shape
        epsilon = torch.randn(*dim).to(self.c.tp.device)
        generated_outputs = epsilon * sigmas + means

        teacher_labels = self.teacher(generated_outputs).float()
        student_labels = self.student(generated_outputs).float()

        if train_gen: # don't compute generator loss when training student for speed
            generated_dist = D.MultivariateNormal(means, torch.diag_embed(sigmas))
            self.coherence = 0.5 * (self.c.tp.generator_loss(teacher_labels, labels_oh) + self.c.tp.generator_loss(labels_oh, teacher_labels))
            self.kl_constraint = D.kl_divergence(generated_dist, self.test_dataset.input_prior).mean()
            self.generator_loss = self.coherence + 10 * self.kl_constraint
        self.student_loss = self.c.tp.student_loss(input=student_labels, target=torch.argmax(teacher_labels, dim=1)) # train student to match teacher
        self.student_acc = (torch.argmax(teacher_labels, dim=1) == torch.argmax(student_labels, dim=1)).float().mean()
        self.teacher_acc = (torch.argmax(teacher_labels, dim=1) == labels).float().mean()
        self.true_student_acc = (torch.argmax(student_labels, dim=1) == labels).float().mean()


        result = {'train/student_loss'     : self.student_loss, 
                  'train/generator_loss'   : self.generator_loss,
                  'train/coherence'        : self.coherence,
                  'train/kl_constraint'    : self.kl_constraint,
                  'train/teacher_acc'      : self.teacher_acc,
                  'train/student_acc'      : self.student_acc,
                  'train/true_student_acc' : self.true_student_acc}

        return result

    def test(self):
        result = defaultdict(lambda: 0)
        with torch.no_grad():
            for i, batch in enumerate(self.test_dataloader):
                data, labels = batch

                student_predictions = torch.argmax(self.student(data), dim=1)
                teacher_predictions = torch.argmax(self.teacher(data), dim=1)

                loss = self.c.tp.test_loss(student_predictions.float(), teacher_predictions.float())
                result['test/loss'] += loss
                result['test/teacher_student_acc']  += (student_predictions == teacher_predictions).float().mean()
                result['test/student_acc'] += (student_predictions == labels).float().mean()
                result['test/teacher_acc'] += (teacher_predictions == labels).float().mean()

        for k in result:
            result[k] = result[k] / (i + 1)

        self.log(result)
        return result



    def zero_grads(self):
        self.teacher.zero_grad()
        self.generator.zero_grad()
        self.student.zero_grad()



    def run(self):
        self.test()
        if not self.load_gen:
            self.total_iterations = len(self.dataloader)*self.c.tp.gen_epochs
            progress_bar = tqdm(total=self.total_iterations)
            generator_loss = torch.tensor(0)
            student_loss = torch.tensor(0)
            for epoch in range(self.c.tp.gen_epochs):
                for index, batch in enumerate(self.dataloader):
                    self.zero_grads()
                    result = self.train(batch)
                    generator_loss = result['train/generator_loss']
                    generator_loss.backward()
                    self.gen_opt.step()

                    if self.iteration % self.c.tp.log_train_every == 0:
                        self.log(result)
                    self.iteration += 1
                    progress_bar.set_description("Epoch {}/{} | Gen Loss {:.2e}".format(epoch+1, self.c.tp.gen_epochs, generator_loss.item()))
                    progress_bar.update(1)


        self.total_iterations = len(self.dataloader)*self.c.tp.stu_epochs
        progress_bar = tqdm(total=self.total_iterations)

        generator_loss = torch.tensor(0)
        student_loss = torch.tensor(0)
        for epoch in range(self.c.tp.stu_epochs):
            for index, batch in enumerate(self.dataloader):
                self.zero_grads()
                result = self.train(batch, train_gen=False)
                student_loss = result['train/student_loss']
                student_loss.backward()
                self.stud_opt.step()

                if self.iteration % self.c.tp.log_train_every == 0:
                    self.log(result)
                self.iteration += 1
                progress_bar.set_description("Epoch {}/{} | Student Loss {:.2e}".format(epoch+1, self.c.tp.stu_epochs, student_loss.item()))
                progress_bar.update(1)
            self.test()
        self.save()


    
    def save(self):
        torch.save(self.generator.state_dict(), os.path.join(self.c.weights_path, '{}.pth'.format('generator')))
        torch.save(self.student.state_dict(), os.path.join(self.c.weights_path, '{}.pth'.format('student')))
        torch.save(self.teacher.state_dict(), os.path.join(self.c.weights_path, '{}.pth'.format('teacher')))