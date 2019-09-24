import logging
import os
import time
import torch
import torch.nn.functional as F

import common
import helpers

# Seeding for reproducibility
torch.manual_seed(common.SEED)

# Set up logging
""" logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(helpers.log_file_path())
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler) """

"""
ADD mkdir for models and logging
"""


class Experiment():  
    def __init__(self, trainloader, testloader, model):
        self.trainloader = trainloader
        self.testloader = testloader
        self.model = model
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
    

    def train(self, criterion, optimizer, epoch, log_interval):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.trainloader):
            
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % log_interval == 0:
                print('Train Epoch: {:3d} [{:6d}/{:6d} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.trainloader.dataset),
                    100. * batch_idx / len(self.trainloader), loss.item()))


    def test(self, criterion, *, monitored=[], reduced_fp=False):
        self.model.eval()
        
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            # Compute test accuracy
            for data, target in self.testloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                    
        test_loss /= len(self.testloader)
        test_accuracy = 100. * correct / len(self.testloader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {:6d}/{:6d} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.testloader.dataset),test_accuracy))

        layer_fp = self.model.compute_fp(monitored)

        layer_reduced_fps = {}
        if reduced_fp:
            layer_reduced_fps = {name: self.model.reduced_fps(name) for name in monitored}

        return test_accuracy, layer_fp, layer_reduced_fps


    def fit(self, epochs, criterion, optimizer, *, monitored=[], save_model=False, log_interval=100):
        initial_acc, initial_fps, _ = self.test(criterion, monitored=monitored)
        test_accuracies = [initial_acc]
        frame_potentials = {name: [fp] for name, fp in initial_fps.items()}

        for epoch in range(1, epochs + 1):
            self.train(criterion, optimizer, epoch, log_interval)
            accuracy, layer_fp, _ = self.test(criterion, monitored=monitored)
            test_accuracies.append(accuracy)
            for name, fp in layer_fp.items():
                frame_potentials[name].append(fp)

        if save_model:
            torch.save(self.model.state_dict(), helpers.model_file_path(self.model.model_ID()))
        
        return test_accuracies, frame_potentials
        