from trainers.trainer import Trainer
from collections import defaultdict 
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os
torch.autograd.set_detect_anomaly(True)

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

    def train_generator(self, batch):
        # generated_input, teacher_output = batch
        # student_output = self.student(generated_input)

        # generator_loss = 0.5 * (self.c.tp.generator_loss(teacher_output, student_output) + self.c.tp.generator_loss(student_output, teacher_output))
        # student_loss = self.c.tp.student_loss(student_output, teacher_output.detach())

        # result = {'train/student_loss': student_loss,
        #           'train/generator_loss': generator_loss}

        # return result

        # label = batch # y
        # generated_image = self.generator(label) # x

        # teacher_label = self.teacher(generated_image)
        # student_label = self.student(generated_image)

        # gen_loss = 0.5 * (self.c.tp.generator_loss(teacher_label, label) + self.c.tp.generator_loss(label, teacher_label)) 
        # stu_loss = self.c.tp.student_loss(student_label, teacher_label.detach())

        # result = {'train/generator_loss': gen_loss, 'train/student_loss': stu_loss}

        # return result


        data, labels = batch
        labels = torch.nn.functional.one_hot(labels).float()
        means, sigmas = self.generator(labels)
        dim = means.shape
        epsilon = torch.randn(self.c.generator.num_samples, *dim).to(self.c.tp.device)
        generated_outputs = epsilon * sigmas + means

        teacher_labels = self.teacher(generated_outputs)
        # student_labels = self.student(generated_outputs)

        generator_loss = 0.5 * (self.c.tp.generator_loss(teacher_labels, torch.cat(self.c.generator.num_samples * [labels.view(1, *labels.shape)], dim=0)) + self.c.tp.generator_loss(torch.cat(self.c.generator.num_samples * [labels.view(1, *labels.shape)], dim=0), teacher_labels)) - 0.0001 * self.c.tp.entropy_loss(means, sigmas)
        # student_loss = -torch.log(self.c.tp.student_loss(student_labels, teacher_labels)) # train student to match teacher

        result = {'train/generator_loss' : generator_loss, 'train/student_loss' : 0}

        return result

    def train_student(self, batch):
        data, labels = batch
        labels = torch.nn.functional.one_hot(labels).float()
        means, sigmas = self.generator(labels)
        dim = means.shape
        epsilon = torch.randn(self.c.generator.num_samples, *dim).to(self.c.tp.device)
        generated_outputs = epsilon * sigmas + means

        teacher_labels = self.teacher(generated_outputs)
        student_labels = self.student(generated_outputs)

        generator_loss = 0.5 * (self.c.tp.generator_loss(teacher_labels, torch.cat(self.c.generator.num_samples * [labels.view(1, *labels.shape)], dim=0)) + self.c.tp.generator_loss(torch.cat(self.c.generator.num_samples * [labels.view(1, *labels.shape)], dim=0), teacher_labels)) - self.c.tp.entropy_loss(means, sigmas)
        student_loss = -torch.log(self.c.tp.student_loss(student_labels, teacher_labels)) # train student to match teacher

        result = {'train/student_loss' : student_loss, 'train/generator_loss' : generator_loss}

        return result

    def test(self):
        # result = defaultdict(lambda: 0)
        # with torch.no_grad():
        #     for i, batch in enumerate(self.test_dataloader):
        #         x,y = batch
        #         pred = self.student(x)
        #         loss = self.c.tp.test_loss(pred, y)
        #         result['test/loss'] += loss
        #         result['test/acc'] += (torch.argmax(pred, dim=1) == y).float().mean()

        # for k in result:
        #     result[k] = result[k]/(i+1)

        # self.log(result)
        # return result

        # result = defaultdict(lambda: 0)
        # with torch.no_grad():
        #     for i, batch in enumerate(self.test_dataloader):
        #         x, y = batch
        #         gen_x = self.generator(y)
        #         pred_student = self.student(gen_x)
        #         pred_teacher = self.teacher(gen_x)
        #         loss = self.c.tp.test_loss(pred_student, pred_teacher)
        #         result['test/loss'] += loss

        # for k in result:
        #     result[k] = result[k] / (i+1)

        # self.log(result)
        # return result

        result = defaultdict(lambda: 0)
        with torch.no_grad():
            for i, batch in enumerate(self.test_dataloader):
                data, labels = batch
                labels = torch.nn.functional.one_hot(labels).float()

                means, sigmas = self.generator(labels)
                dim = means.shape
                epsilon = torch.randn(*dim).to(self.c.tp.device)
                generated_outputs = epsilon * sigmas + means

                student_predictions = self.student(generated_outputs)
                teacher_predictions = self.teacher(generated_outputs)

                loss = self.c.tp.test_loss(student_predictions, teacher_predictions)
                result['test/loss'] += loss
                result['test/acc']  += (torch.argmax(student_predictions, dim=1) == torch.argmax(teacher_predictions, dim=1)).float().mean()

        for k in result:
            result[k] = result[k] / (i + 1)

        self.log(result)
        return result



    def zero_grads(self):
        self.teacher.zero_grad()
        self.generator.zero_grad()
        self.student.zero_grad()



    def run(self):
        self.total_iterations = len(self.dataloader)*self.c.tp.gen_epochs
        progress_bar = tqdm(total=self.total_iterations)
        
        self.test()
        generator_loss = torch.tensor(0)
        student_loss = torch.tensor(0)
        for epoch in range(self.c.tp.gen_epochs):
            for index, batch in enumerate(self.dataloader):
                self.zero_grads()
                result = self.train_generator(batch)
                generator_loss = result['train/generator_loss']
                generator_loss.backward()
                self.gen_opt.step()

                if self.iteration % self.c.tp.log_train_every == 0:
                    self.log(result)
                self.iteration += 1
                progress_bar.set_description("Epoch {}/{} | Gen Loss {:.2e}".format(epoch+1, self.c.tp.epochs, generator_loss.item()))
                progress_bar.update(1)


        self.total_iterations = len(self.dataloader)*self.c.tp.stu_epochs
        progress_bar = tqdm(total=self.total_iterations)

        generator_loss = torch.tensor(0)
        student_loss = torch.tensor(0)
        for epoch in range(self.c.tp.stu_epochs):
            for index, batch in enumerate(self.dataloader):
                self.zero_grads()
                result = self.train_student(batch)
                student_loss = result['train/student_loss']
                student_loss.backward()
                self.stud_opt.step()

                if self.iteration % self.c.tp.log_train_every == 0:
                    self.log(result)
                self.iteration += 1
                progress_bar.set_description("Epoch {}/{} | Gen Loss {:.2e}".format(epoch+1, self.c.tp.epochs, generator_loss.item()))
                progress_bar.update(1)
            self.test()
        self.save()


    
    def save(self):
        torch.save(self.generator.state_dict(), os.path.join(self.c.weights_path, '{}.pth'.format('generator')))
        torch.save(self.student.state_dict(), os.path.join(self.c.weights_path, '{}.pth'.format('student')))
        torch.save(self.teacher.state_dict(), os.path.join(self.c.weights_path, '{}.pth'.format('teacher')))