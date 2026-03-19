import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(123)

class MLP(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 20),
            nn.ReLU(),

            nn.Linear(20, 10),
            nn.ReLU(),

            nn.Linear(10, num_outputs)           
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

model = MLP(num_inputs=2, num_outputs=2)
print(model)

print(model.layers[0].weight.shape)
print("CUDA available:", torch.cuda.is_available())

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_params}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])

y_train = torch.tensor([0, 0, 0, 1, 1])

X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
])

y_test = torch.tensor([0, 1])

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
train_dataset = SimpleDataset(X_train, y_train)
test_dataset = SimpleDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    drop_last=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
num_epochs = 4

for epoch in range(num_epochs):

    model.train()

    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):

        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        logits = model(x_batch)
        loss = F.cross_entropy(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1:03d}/{num_epochs:03d}, Batch {batch_idx:03d}/{len(train_loader):03d}, Loss: {loss.item():.4f}")


def compute_accuracy(model, dataloader):

    model.eval()
    correct = 0.0
    total_examples = 0
    
    for idx, (features, labels) in enumerate(dataloader):
        
        with torch.no_grad():
            logits = model(features)
        
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return (correct / total_examples).item()

print(f"Test Accuracy: {compute_accuracy(model, test_loader):.4f}")

#torch.save(model.state_dict(), "model.pt")
#model = MLP(2, 2) # needs to match the original model exactly
#model.load_state_dict(torch.load("model.pt", weights_only=True))