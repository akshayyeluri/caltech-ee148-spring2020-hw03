from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler

import matplotlib.pyplot as plt
import os
import json

'''
This code is adapted from two sources:
(i) The official PyTorch MNIST example (https://github.com/pytorch/examples/blob/master/mnist/main.py)
(ii) Starter code from Yisong Yue's CS 155 Course (http://www.yisongyue.com/courses/cs155/2020_winter/)
'''

class fcNet(nn.Module):
    '''
    Design your model with fully connected layers (convolutional layers are not
    allowed here). Initial model is designed to have a poor performance. These
    are the sample units you can try:
        Linear, Dropout, activation layers (ReLU, softmax)
    '''
    def __init__(self):
        # Define the units that you will use in your model
        # Note that this has nothing to do with the order in which operations
        # are applied - that is defined in the forward function below.
        super(fcNet, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=20)
        self.fc2 = nn.Linear(20, 10)
        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Define the sequence of operations your model will apply to an input x
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.relu(x)

        output = F.log_softmax(x, dim=1)
        return output


class ConvNet(nn.Module):
    '''
    Final model
    '''
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5,5))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5))
        self.bn = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout2d(0.3)
        self.fc1 = nn.Linear(4 * 4 * 64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = self.bn(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output



class ConvNetV0(nn.Module):
    '''
    Design your model with convolutional layers.
    '''
    def __init__(self):
        super(ConvNetV0, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(8, 8, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(200, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()   # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output = model(data)                # Make predictions
        loss = F.nll_loss(output, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, validating=False, ret_all=False):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_num += len(data)

    test_loss /= test_num

    if validating:
        print('\n Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, test_num,
            100. * correct / test_num))
    # Return either loss or (loss, accuracy)
    return (test_loss, (correct / test_num)) if ret_all else test_loss


def main(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    print(args)
    transform_lst = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])


    # Pytorch has default MNIST dataloader which loads data at each iteration
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                transform=transform_lst)


    # Evaluate on the official test set
    if args.evaluate:
        assert os.path.exists(args.load_model)

        # Set the test model
        model = ConvNetV0().to(device)
        model.load_state_dict(torch.load(args.load_model))

        test_dataset = datasets.MNIST('../data', train=False,
                    transform=transform_lst)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        train_loss, train_acc = test(model, device, train_loader, ret_all=True)
        print('Train loss={:.4f}, accuracy={:.1f}'.format(train_loss, 100 * train_acc))

        test_loss, test_acc = test(model, device, test_loader, ret_all=True)
        print('Test loss={:.4f}, accuracy={:.1f}'.format(test_loss, 100 * test_acc))
        return train_loss, train_acc, test_loss, test_acc


    
    # Save random validation_frac for validation, and random train_frac
    # for training
    n = len(train_dataset)
    val_size = int(args.validation_frac * n)
    train_size = min(int(args.train_frac * n), n - val_size)
    perm = np.random.permutation(n)
    subset_indices_train = perm[val_size:val_size + train_size]
    subset_indices_valid = perm[:val_size]

    #import IPython; IPython.embed()

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(subset_indices_train)
    )
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.test_batch_size,
        sampler=SubsetRandomSampler(subset_indices_valid)
    )

    # Load your model [fcNet, ConvNetV0, Net]
    model = ConvNetV0().to(device)

    # Try different optimzers here [Adam, SGD, RMSprop]
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Set your learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    train_losses = np.empty(args.epochs)
    valid_losses = np.empty(args.epochs)
    # Training loop
    for i, epoch in enumerate(range(1, args.epochs + 1)):
        train(args, model, device, train_loader, optimizer, epoch)
        train_losses[i] = test(model, device, train_loader)
        valid_losses[i] = test(model, device, val_loader, validating=True)
        scheduler.step()    # learning rate scheduler

    if args.save_model:
        torch.save(model.state_dict(), "mnist_model.pt")

    if args.learning_curves:
        xs = np.arange(1, args.epochs + 1)
        plt.plot(xs, train_losses, label='Training loss')
        plt.plot(xs, valid_losses, label='Validation loss')
        plt.xlabel('Epoch'); plt.ylabel('Negative Log Likelihood Loss');
        plt.legend();
        plt.savefig('learning_curves.png')

        
def parse_arguments():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate your model on the official test set')
    parser.add_argument('--load-model', type=str,
                        help='model file path')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')

    # New things I added
    parser.add_argument('--learning-curves', action='store_true', default=True,
                        help='Generate learning curves')
    parser.add_argument('--validation-frac', type=float, default=0.15, 
                        help='What fraction of training set to use for validation')
    parser.add_argument('--train-frac', type=float, default = 1.0,
                        help='What size subset of training set to use for training')
    parser.add_argument('--save-to-file', action='store_true', default=False,
                        help='Whether or not to save the evaluate results to a file')
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_arguments()
    save_fname = "train_subset.json"

    res = main(args)
    if args.evaluate and args.save_to_file:
        try:
            with open(save_fname, 'r') as f:
                subset_data = json.load(f)
        except FileNotFoundError as e:
            subset_data = {}

        label = args.load_model
        label = label[label.find('_') + 1: -3] # leave out .pt
        subset_data[label] = res
        with open(save_fname, 'w') as f:
            json.dump(subset_data, f)





