import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from urllib.request import urlretrieve
import gzip

# datasets URLs
BASE_URL = 'https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion'
RESOURCES = {
    'train': ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz'],
    'test': ['t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
}

# Download dataset files if they do not exist
def download_fashion_mnist(data_dir='./data'):
    os.makedirs(data_dir, exist_ok=True)
    for url in [TRAIN_IMAGES_URL, TRAIN_LABELS_URL, TEST_IMAGES_URL, TEST_LABELS_URL]:
        filename = os.path.join(data_dir, url.split('/')[-1])
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            urlretrieve(url, filename)

# Load dataset from gzip files
def load_fashion_mnist(data_dir='./data', train=True):
    if train:
        images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
    else:
        images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
    
    with gzip.open(images_path, 'rb') as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    
    with gzip.open(labels_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    
    return images, labels

# Custom dataset class 
class CustomFashionMNIST(Dataset):
    def __init__(self, root='./data', train=True, transform=None):
        download_fashion_mnist(root)
        self.data, self.targets = load_fashion_mnist(root, train)
        self.data = self.data.astype(np.float32) / 255.0
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        image = torch.FloatTensor(image).unsqueeze(0)
        if self.transform:
            image = self.transform(image)
        return image, label

# CNN architecture
class FashionNet(nn.Module):
    def __init__(self):
        super(FashionNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Save model weights and training 
def save_model(model, optimizer, epoch, accuracy, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy
    }
    last_filename = 'fashion_model_last.pt'
    torch.save(checkpoint, last_filename)
    print(f"Last model saved to {last_filename}")
    
    if is_best:
        best_filename = 'fashion_model_best.pt'
        torch.save(checkpoint, best_filename)
        print(f"Best model saved to {best_filename}")

# Load model weights
def load_model(filename='fashion_model_best.pt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FashionNet().to(device)
    
    try:
        checkpoint = torch.load(filename, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        accuracy = checkpoint.get('accuracy', 0.0)
        epoch = checkpoint.get('epoch', 0)
        print(f"Model loaded from {filename}")
        print(f"Accuracy: {accuracy:.2f}%")
        return model, accuracy, epoch
    except FileNotFoundError:
        print(f"No saved model found at {filename}")
        return model, 0.0, 0

# Evaluate model on test dataset
def evaluate_model(model_path='fashion_model_best.pt'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        model, saved_accuracy, last_epoch = load_model(model_path)
        model.eval()

        test_dataset = CustomFashionMNIST(train=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        correct = 0
        total = 0

        print(f"\nEvaluating model from epoch {last_epoch}")
        print(f"Saved accuracy: {saved_accuracy:.2f}%")

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        current_accuracy = 100 * correct / total
        print(f'Current Evaluation Accuracy: {current_accuracy:.2f}%')

        # sample predictions
        print("Sample Predictions:")
        num_samples = 5
        for i in range(num_samples):
            sample_image, sample_label = test_dataset[i]
            sample_image = sample_image.unsqueeze(0).to(device)
            prediction = model(sample_image).argmax(1)[0].item()
            print(f"Sample {i + 1}: Predicted = {prediction}, Actual = {sample_label}")

    except FileNotFoundError:
        print(f"Error: Could not find model file '{model_path}'")
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")

# training and evaluation 
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create datasets and loaders
    train_dataset = CustomFashionMNIST(train=True)
    test_dataset = CustomFashionMNIST(train=False)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Create model
    model = FashionNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    num_epochs = 10
    best_accuracy = 0.0
    print("Starting training")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        print(f'Epoch: {epoch}, Accuracy: {accuracy:.2f}%')

        # Save both best and last models
        is_best = accuracy > best_accuracy
        if is_best:
            best_accuracy = accuracy
            print(f'New best accuracy: {accuracy:.2f}%')
        save_model(model, optimizer, epoch, accuracy, is_best=is_best)
        
    print(f"\nTraining completed!")
    print(f"Best accuracy achieved: {best_accuracy:.2f}%")

    print("Evaluating Best Model:")
    evaluate_model('fashion_model_best.pt')

    print("Evaluating Last Model:")
    evaluate_model('fashion_model_last.pt')
