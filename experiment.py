import numpy as np
import logging
import os
import time
import torch
import torch.nn.functional as f

import common
import helpers

# Seeding for reproducibility
np.random.seed(common.SEED)
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

def weight_properties(w):
    rows, cols = w.shape
    w_normalized = f.normalize(w, p=2, dim=1)
    #if rows < cols:
    T_mod = torch.mm(w_normalized, w_normalized.T)  # changeme?
    #else:
    #    T_mod = torch.mm(w.T, w)

    #weight_norm_sum = T_mod.trace().item()
    inner_product_sum = torch.mm(T_mod, T_mod).trace().item()
    
    return inner_product_sum / (rows**2) 


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


    def test(self, criterion, test_accuracies, frame_potentials):
        self.model.eval()
        
        test_loss = 0
        correct = 0
        frame_potential = {}
        
        with torch.no_grad():
            # Compute test accuracy
            for data, target in self.testloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max probability
                correct += pred.eq(target.view_as(pred)).sum().item()

            # Compute frame potentials
            for name, param in self.model.named_parameters():
                if name in frame_potentials.keys():
                    normalized_fp = weight_properties(param.data)
                    #norm_sums[name].append(weight_norm_sum)
                    frame_potentials[name].append(normalized_fp)
                    

        test_loss /= len(self.testloader)
        test_accuracy = 100. * correct / len(self.testloader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {:6d}/{:6d} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.testloader.dataset),test_accuracy))

        test_accuracies.append(test_accuracy)


    def fit(self, epochs, criterion, optimizer, *, monitored=[], log_interval=100):
        test_accuracies = []
        frame_potentials = {layer: [] for layer in monitored}
        #norm_sums = {layer: [] for layer in monitored}
        
        self.test(criterion, test_accuracies, frame_potentials)

        for epoch in range(1, epochs + 1):
            self.train(criterion, optimizer, epoch, log_interval)
            self.test(criterion, test_accuracies, frame_potentials)
        
        return test_accuracies, frame_potentials
        