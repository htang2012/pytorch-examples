import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# Load the dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training loop (for simplicity, training only for one epoch)
model.train()
for epoch in range(1):
    print(f"epoch: {epoch} ")
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

import numpy as np

# Set the model to evaluation mode
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

# Convert to tensors
all_preds = torch.tensor(all_preds)
all_labels = torch.tensor(all_labels)

# Compute the confusion matrix
num_classes = 10
conf_matrix = torch.zeros(num_classes, num_classes)
for t, p in zip(all_labels, all_preds):
    conf_matrix[t.long(), p.long()] += 1

print("Confusion Matrix:\n", conf_matrix)

# Extract TP, FP, FN, and TN for each class
TP = torch.diag(conf_matrix)
FP = conf_matrix.sum(dim=0) - TP
FN = conf_matrix.sum(dim=1) - TP
TN = conf_matrix.sum() - (FP + FN + TP)

# Compute metrics
accuracy = (TP.sum() / conf_matrix.sum()).item()
precision = (TP / (TP + FP)).mean().item()
recall = (TP / (TP + FN)).mean().item()
f1_score = (2 * precision * recall) / (precision + recall)
specificity = (TN / (TN + FP)).mean().item()
balanced_accuracy = (recall + specificity) / 2  # no .item() needed here

# Display results
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"Balanced Accuracy: {balanced_accuracy:.2f}")

import matplotlib.pyplot as plt
import seaborn as sns

# Convert the confusion matrix to integer for plotting
conf_matrix = conf_matrix.int()

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix.numpy(), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

