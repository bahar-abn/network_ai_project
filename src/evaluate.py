import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from src.models import FCModel
from sklearn.metrics import accuracy_score, confusion_matrix


df = pd.read_csv('data/traffic_features.csv')
X = torch.tensor(df.drop(columns=[]).values, dtype=torch.float32)
y = torch.tensor((df['protocol'] % 2).values, dtype=torch.long)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=16)


input_dim = X.shape[1]
model = FCModel(input_dim)
model.load_state_dict(torch.load('model.pth'))
model.eval()

preds = []
labels = []
with torch.no_grad():
    for batch_X, batch_y in loader:
        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        preds.extend(predicted.numpy())
        labels.extend(batch_y.numpy())

acc = accuracy_score(labels, preds)
cm = confusion_matrix(labels, preds)
print("Accuracy:", acc)
print("Confusion Matrix:\n", cm)
