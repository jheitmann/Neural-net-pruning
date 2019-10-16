import numpy as np
import os
import time
import torch
import torch.nn.functional as F

import common
import helpers

# Seeding for reproducibility, only once (remove from nb?)
torch.manual_seed(common.SEED)

"""
def save_training_meta(model, epochs, test_accuracies, frame_potentials, *, inner_products={}):
    model_fname = helpers.model_file_path(model.model_id())
    torch.save(model.state_dict(), model_fname)
    print("Saved trained model to:", model_fname)
    acc_fname = helpers.train_results_path(model.model_id(), epochs, "acc")
    np.save(acc_fname, test_accuracies)
    print("Saved validation accuracies to:", acc_fname + ".npy")
    if frame_potentials:
        fp_fname = helpers.train_results_path(model.model_id(), epochs, "fp")
        np.save(fp_fname, frame_potentials)
        print("Saved frame potentials to:", fp_fname + ".npy")
    if inner_products:
        for layer, ips_ls in inner_products.items():
            ips_fname = helpers.train_results_path(model.model_id(), epochs, "ips", layer=layer)
            np.save(ips_fname, torch.stack(ips_ls).numpy())
            print("Saved {} cosine distances to: {}".format(layer, ips_fname + ".npy"))
"""


class Experiment:
    def __init__(self, model, criterion, optimizer, optim_kwargs, *, model_state={}):
        self.model = model
        self.model_state = model_state
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.optim_kwargs = optim_kwargs

    def init_model(self):
        self.model.load_state_dict(self.model_state)
        self.model.to(self.device)

    def clone(self):
        Model = self.model.__class__
        new_model = Model()
        new_model_state = self.model.state_dict()
        new_model.load_state_dict(new_model_state)

        Optimizer = self.optimizer.__class__
        new_optimizer = Optimizer(new_model.parameters(), **self.optim_kwargs)

        return Experiment(new_model, self.criterion, new_optimizer,
                          model_state=new_model_state, optim_kwargs=self.optim_kwargs)

    def batch_train(self, data, target):
        self.model.train()
        data, target = data.to(self.device), target.to(self.device)

        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, trainloader, epoch, log_interval):
        for batch_idx, (data, target) in enumerate(trainloader):
            train_loss = self.batch_train(data, target)

            if batch_idx % log_interval == 0:
                print("Train Epoch: {:3d} [{:6d}/{:6d} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch, batch_idx * len(data), len(trainloader.dataset),
                           100. * batch_idx / len(trainloader), train_loss))

    def test(self, testloader, monitored):  # changeme: monitored not needed
        self.model.eval()

        test_loss = 0
        correct = 0

        with torch.no_grad():
            # Compute test accuracy
            for data, target in testloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(testloader)
        test_accuracy = 100. * correct / len(testloader.dataset)

        layer_ips, layer_fp = self.model.compute_fp(monitored)

        print("\nTest set: Average loss: {:.4f}, Accuracy: {:6d}/{:6d} ({:.0f}%)\n".format(
            test_loss, correct, len(testloader.dataset), test_accuracy))

        return test_accuracy, layer_ips, layer_fp

    def fit(self, trainloader, testloader, epochs, *, monitored=[], save_results=False, log_interval=100):
        initial_acc, initial_ips, initial_fps = self.test(testloader, monitored)
        test_accuracies = [initial_acc]

        if save_results:
            dir_name = helpers.model_file_path(self.model.model_id())
            os.mkdir(dir_name)
            print("Created model snapshot directory:", dir_name)
            snapshot_fname = os.path.join(dir_name, '0')
            torch.save(self.model.state_dict(), snapshot_fname)

        for epoch in range(1, epochs + 1):
            self.train(trainloader, epoch, log_interval)
            accuracy, layer_ips, layer_fp = self.test(testloader, monitored)
            test_accuracies.append(accuracy)
            if save_results:
                snapshot_fname = os.path.join(dir_name, str(epoch))
                torch.save(self.model.state_dict(), snapshot_fname)

        if save_results:
            print("Saved final model to:", snapshot_fname)

        return test_accuracies
