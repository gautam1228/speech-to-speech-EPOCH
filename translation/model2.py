import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define the dataset class
class SpeechDataset(Dataset):
    def __init__(self, english_data, target_data):
        self.english_data = english_data
        self.target_data = target_data
        
    def __len__(self):
        return len(self.english_data)
    
    def __getitem__(self, idx):
        return self.english_data[idx], self.target_data[idx]

# Define the sequence-to-sequence model
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2Seq, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_seq):
        hidden = torch.relu(self.encoder(input_seq))
        output_seq = self.decoder(hidden)
        return output_seq

# Hyperparameters
input_size = 100
hidden_size = 512
output_size = 100
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# Create dataset and dataloaders
english_data = [np.random.randn(100) for _ in range(5000)] 
target_data = [np.random.randn(100) for _ in range(5000)] 
# print(english_data)
dataset = SpeechDataset(english_data, target_data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
model = Seq2Seq(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (english_batch, target_batch) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Convert numpy arrays to PyTorch tensors with float32 dtype
        english_batch = torch.tensor(english_batch, dtype=torch.float32)
        target_batch = torch.tensor(target_batch, dtype=torch.float32)
        
        # Forward pass
        output_batch = model(english_batch)
        
        # Compute loss
        loss = criterion(output_batch, target_batch)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}')

print('Training finished.')
# Save the trained model
model_path = "C:\\Users\\mehul\\Downloads\\epoch\\speech-to-speech-EPOCH\\translation\\model_new.pth"
torch.save(model.state_dict(), model_path)
