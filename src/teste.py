import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

# Configuração do SummaryWriter

# Definir um modelo simples
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Carregar dataset

def f():
    writer = SummaryWriter('runs/experimento2')

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Inicializar modelo, critério de perda e otimizador
    model = SimpleModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

# Treinamento
    for epoch in range(2):  # número de épocas
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            # Zerar gradientes
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass e otimização
            loss.backward()
            optimizer.step()

            # Atualizar perda acumulada
            running_loss += loss.item()
            
            if i % 100 == 99:  # Log a cada 100 mini-batches
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100}')
                writer.add_scalar('training loss', running_loss / 100, epoch * len(trainloader) + i)
                running_loss = 0.0

    print('Treinamento finalizado')
    writer.close()

# Fechar o SummaryWriter
f()

