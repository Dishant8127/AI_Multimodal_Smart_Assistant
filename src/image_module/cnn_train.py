# Training skeleton for transfer learning with PyTorch
import torch
import torchvision
from torchvision import models, transforms, datasets
from torch import nn, optim

def train_transfer_learning(data_dir, epochs=3, batch_size=16, lr=1e-4, device='cpu'):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    train_ds = datasets.ImageFolder(data_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(train_ds.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs} loss: {running_loss/len(train_loader)}')

    torch.save(model.state_dict(), 'models/cnn/cnn_model.pth')
    print('Model saved to models/cnn/cnn_model.pth')
