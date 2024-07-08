import os
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

data_dir = 'Dataset'

classes = ['Very_Mild_Demented', 'Non_Demented', 'Moderate_Demented', 'Mild_Demented']

def load_and_display_images(data_dir, classes):
    fig, axes = plt.subplots(1, len(classes), figsize=(15, 5))
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, 'train', class_name)
        image_files = os.listdir(class_dir)
        
        if not image_files:
            print(f"Görseller bulunamadı: {class_dir}")
            continue
        
        image_path = os.path.join(class_dir, image_files[0])
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Görseller Yüklenemedi: {image_path}")
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axes[i].imshow(image)
        axes[i].set_title(class_name)
        axes[i].axis('off')
    plt.show()

load_and_display_images(data_dir, classes)

image_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

train_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=image_transforms['train'])
test_dataset = ImageFolder(os.path.join(data_dir, 'test'), transform=image_transforms['test'])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f'\nEğitim Veri Seti Boyutu: {len(train_dataset)}')
print(f'Test Veri Seti Boyutu: {len(test_dataset)}')

class MRI_CNN(nn.Module):
    def __init__(self, num_classes=4):
        super(MRI_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MRI_CNN(num_classes = len(classes))
print('Model Eğitime Hazır.')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch: {epoch + 1} / {num_epochs}, Loss: {epoch_loss:.2f}')
    
print('Model Eğitimi Tamamlandı.')

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
conf_matrix = confusion_matrix(all_labels, all_preds)

print(f'Model Doğruluk Oranı: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)

plt.plot(range(num_epochs), epoch_loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Eğitim Kayıpları')
plt.show()

def visualize_predictions(data_loader, model, num_images=6):
    model.eval()
    images_so_far = 0
    fig, axes = plt.subplots(1, num_images, figsize=(25, 5))
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for j in range(inputs.size()[0]):
                if images_so_far >= num_images:
                    return
                ax = axes[images_so_far]
                img = inputs[j].cpu().numpy().transpose((1, 2, 0))
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)
                ax.imshow(img)
                ax.set_title(f'Tahmin: {classes[preds[j]]}\nGerçek: {classes[labels[j]]}')
                ax.axis('off')
                images_so_far += 1
        visualize_predictions(test_loader, model)