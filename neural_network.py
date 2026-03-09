import numpy as np
from urllib import request
import gzip
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def load_mnist():
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images":  "t10k-images-idx3-ubyte.gz",
        "test_labels":  "t10k-labels-idx1-ubyte.gz"
    }

    os.makedirs("data", exist_ok=True)

    for key, filename in files.items():
        path = f"data/{filename}"
        if not os.path.exists(path):
            print(f"Downloading {filename}...")
            request.urlretrieve(base_url + filename, path)

    # Charger images train
    with gzip.open("data/train-images-idx3-ubyte.gz", "rb") as f:
        train_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)

    # Charger labels train
    with gzip.open("data/train-labels-idx1-ubyte.gz", "rb") as f:
        train_labels = np.frombuffer(f.read(), np.uint8, offset=8)

    # Charger images test
    with gzip.open("data/t10k-images-idx3-ubyte.gz", "rb") as f:
        test_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)

    # Charger labels test
    with gzip.open("data/t10k-labels-idx1-ubyte.gz", "rb") as f:
        test_labels = np.frombuffer(f.read(), np.uint8, offset=8)

    return train_images, train_labels, test_images, test_labels

class MNISTNet(nn.Module):
      def __init__(self):
          super().__init__()
          # 3 couches comme ton réseau from scratch
          # 784 -> 128 -> 64 -> 10
          self.layer1 = nn.Linear(784, 128)
          self.layer2 = nn.Linear(128, 64)
          self.layer3 = nn.Linear(64, 10)
          self.relu   = nn.ReLU()

      def forward(self, x):
          # Couche 1
          x = self.relu(self.layer1(x))
          # Couche 2
          x = self.relu(self.layer2(x))
          # Couche 3
          x = self.layer3(x)
          return x

model = MNISTNet()
print(model)

  # Test avec un batch de 32 images
x = torch.randn(32, 784)
output = model(x)
print(f"Output shape: {output.shape}")  # attendu : (32, 10)


train_images, train_labels, test_images, test_labels = load_mnist()
train_images = train_images / 255.0
test_images  = test_images / 255.0

# Convertir les données NumPy en tenseurs PyTorch
X_train = torch.tensor(train_images, dtype=torch.float32)
Y_train = torch.tensor(train_labels, dtype=torch.long)
X_test  = torch.tensor(test_images, dtype=torch.float32)
Y_test  = torch.tensor(test_labels, dtype=torch.long)

# Dataset et DataLoader - gère les batches automatiquement
train_dataset = TensorDataset(X_train, Y_train)
train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Modèle, loss, optimizer
model     = MNISTNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Boucle d'entraînement
for epoch in range(20):
    epoch_loss = 0
    for X_batch, Y_batch in train_loader:
        # 1. Forward pass
        output = model(X_batch)

        # 2. Calcul de la loss
        loss = criterion(output, Y_batch)

        # 3. Reset des gradients
        optimizer.zero_grad()

        # 4. Backward pass
        loss.backward()

        # 5. Mise à jour des poids
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/20, Loss: {epoch_loss/len(train_loader):.4f}")

# Evaluation
with torch.no_grad():
    output = model(X_test)
    pred   = torch.argmax(output, dim=1)
    acc    = (pred == Y_test).float().mean()
    print(f"Test accuracy: {acc:.4f}")

'''
Avant de lancer — fais le mapping avec ton code from scratch :

optimizer.zero_grad()  ->  initialisation des gradients à zéro
loss.backward()        ->  calcul du bakward (backpropagation)
optimizer.step()       ->  mise à jour des poids
'''
