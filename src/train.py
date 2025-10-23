import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from src.models import FCModel, CNNModel, SimpleTransformer


df = pd.read_csv('data/traffic_features.csv')
X = df.drop(columns=[]).values 
y = (df['protocol'] % 2).values  

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)


input_dim = X.shape[1]
model = FCModel(input_dim) 


criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)


for epoch in range(5):  
    running_loss = 0.0
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(loader):.4f}")

torch.save(model.state_dict(), 'model.pth')
print("Model trained and saved!")
