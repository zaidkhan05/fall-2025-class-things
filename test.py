#pytorch base test file
import torch
import torch.nn as nn
import torch.optim as optim
import unittest
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class TestSimpleNN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load MNIST dataset
        transform = transforms.Compose([transforms.ToTensor()])
        mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
        cls.train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
        cls.test_loader = DataLoader(mnist_test, batch_size=1000, shuffle=False)
        
        # Initialize model, loss function and optimizer
        cls.model = SimpleNN()
        cls.criterion = nn.CrossEntropyLoss()
        cls.optimizer = optim.SGD(cls.model.parameters(), lr=0.01, momentum=0.9)

    def test_training_step(self):
        self.model.train()
        data_iter = iter(self.train_loader)
        images, labels = next(data_iter)
        
        # Forward pass
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        
        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)

    def test_evaluation_step(self):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        self.assertGreater(accuracy, 50)  # Expect at least 50% accuracy
if __name__ == '__main__':
    unittest.main()

