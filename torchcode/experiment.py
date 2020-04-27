import json
import numpy as np
import os
import torch

import common
import torchcode.helpers as helpers

# Seeding for reproducibility, only once (remove from nb?)
torch.manual_seed(common.SEED)


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
        model_class = self.model.__class__
        new_model = model_class()
        new_model_state = self.model.state_dict()
        new_model.load_state_dict(new_model_state)

        if self.optimizer:
            optimizer_class = self.optimizer.__class__
            new_optimizer = optimizer_class(new_model.parameters(), **self.optim_kwargs)
        else:
            new_optimizer = None

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

    def test(self, testloader, monitored):
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

        layer_fp = {}  # self.model.compute_fp(monitored)

        print("\nTest set: Average loss: {:.4f}, Accuracy: {:6d}/{:6d} ({:.0f}%)\n".format(
            test_loss, correct, len(testloader.dataset), test_accuracy))

        return test_accuracy, layer_fp

    # Might add learning-rate schedule
    def fit(self, trainloader, testloader, epochs, *, monitored=[],
            log_interval=100, save_results=False, save_interval=1):

        initial_acc, initial_fps = self.test(testloader, monitored)
        test_accuracies = [initial_acc]
        base_dir = ""

        if save_results:
            base_dir, dir_paths = helpers.model_results_path(self.model.model_id())
            snapshot_dir = dir_paths[common.SNAPSHOTS_DIR]
            snapshot_fname = os.path.join(snapshot_dir, '0')
            torch.save(self.model.state_dict(), snapshot_fname)

        for epoch in range(1, epochs + 1):
            self.train(trainloader, epoch, log_interval)
            accuracy, layer_fp = self.test(testloader, monitored)
            test_accuracies.append(accuracy)
            if (epoch % save_interval == 0) and save_results:
                snapshot_fname = os.path.join(snapshot_dir, str(epoch // save_interval))
                torch.save(self.model.state_dict(), snapshot_fname)

        if save_results:
            print("Saved final model to:", snapshot_fname)
            training_dir = dir_paths[common.TRAINING_DIR]
            acc_path = os.path.join(training_dir, common.ACCURACY_FNAME)
            np.save(acc_path, test_accuracies)
            print("Saved validation accuracies to:", acc_path)
            if os.path.exists(common.MODEL_SPECS_PATH):
                with open(common.MODEL_SPECS_PATH, 'r') as rfp:
                    models = json.load(rfp)
            else:
                models = {}
            layers = {name.split('.')[0] for name, _ in self.model.named_parameters()}
            models[base_dir] = list(layers)
            with open(common.MODEL_SPECS_PATH, 'w') as wfp:
                json.dump(models, wfp, sort_keys=True, indent=4)

        return test_accuracies, base_dir
