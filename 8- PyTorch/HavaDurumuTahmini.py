import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class HavaDurumuCNN(nn.Module):
    def __init__(self):
        super(HavaDurumuCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flattened_size = self._get_flattened_size()
        
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, 11)

    def _get_flattened_size(self):
        x = torch.zeros(1, 3, 128, 128)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.flattened_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.ImageFolder(root = 'dataset/train', transform=transform)
test_dataset = datasets.ImageFolder(root = 'dataset/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = HavaDurumuCNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(100 * correct / total)

    model.eval()
    running_loss2, correct2, total2 = 0.0, 0, 0
    with torch.no_grad():
        for images2, labels2 in test_loader:
            outputs2 = model(images2)
            loss2 = criterion(outputs2, labels2)

            running_loss2 += loss.item()
            _, predicted2 = torch.max(outputs, 1)
            total2 += labels.size(0)
            correct2 = (predicted2 == labels).sum().item()
    test_losses.append(running_loss2 / len(test_loader))
    test_accuracies.append(100 * correct2 / total2)

    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Eğitim Kaybı: {train_losses[-1]:.4f}, Eğitim Doğruluğu: {train_accuracies[-1]:.2f}%, "
          f"Test Kaybı: {test_losses[-1]:.4f}, Test Doğruluğu: {test_accuracies[-1]:.2f}%")
    
plt.figure(figsize = (10, 5))
plt.plot(range(num_epochs), train_losses, label = 'Eğitim Kaybı')
plt.plot(range(num_epochs), test_losses, label = 'Test Kaybı')
plt.xlabel('Epochs')
plt.ylabel('Kayıplar')
plt.legend()
plt.title('Eğitim ve Test Kayıpları')

plt.figure(figsize = (10, 5))
plt.plot(range(num_epochs), train_accuracies, label = 'Eğitim Doğruluğu')
plt.plot(range(num_epochs), test_accuracies, label = 'Test Doğruluğu')
plt.xlabel('Epochs')
plt.ylabel('Doğruluk (%)')
plt.legend()
plt.title('Eğitim ve Test Doğrulukları')

plt.show()