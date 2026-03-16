import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def rnn_step(x_t, h_prev, W_x, W_h, b):
  return np.tanh(W_x @ x_t + W_h @ h_prev + b)

def rnn_forward(inputs, h0, W_x, W_h, b):
    seq_len = len(inputs)
    hidden_size = len(h0)
    
    hidden_states = np.zeros((seq_len, hidden_size))
    h_t = h0
    
    for t in range(seq_len):
        h_t = rnn_step(inputs[t], h_t, W_x, W_h, b)
        hidden_states[t] = h_t
    
    return hidden_states

def lstm_step(x_t, h_prev, c_prev, W_f, W_i, W_g, W_o, b_f, b_i, b_g, b_o):
    
    combined = np.concatenate([h_prev, x_t])
    

    f_t = sigmoid(W_f @ combined + b_f)
    i_t = sigmoid(W_i @ combined + b_i) 
    g_t = np.tanh(W_g @ combined + b_g)
    o_t = sigmoid(W_o @ combined + b_o)
    
    c_t = f_t * c_prev + i_t * g_t
    h_t = o_t * np.tanh(c_t) 
    
    return h_t, c_t

def lstm_forward(inputs, h0, c0, W_f, W_i, W_g, W_o, b_f, b_i, b_g, b_o):
  seq_len = len(inputs)
  hidden_size = len(h0)
      
  hidden_states = np.zeros((seq_len, hidden_size))
  h_t = h0
  c_t = c0

  for t in range(seq_len):
        h_t, c_t = lstm_step(inputs[t], h_t, c_t, W_f, W_i, W_g, W_o, b_f, b_i, b_g, b_o)
        hidden_states[t] = h_t
      
  return hidden_states

import torch
import torch.nn as nn

lstm = nn.LSTM(input_size=4, hidden_size=3, batch_first=True)

x  = torch.randn(1, 5, 4) 
h0 = torch.zeros(1, 1, 3) 
c0 = torch.zeros(1, 1, 3)

output, (h_n, c_n) = lstm(x, (h0, c0))
print(f"output shape: {output.shape}") 
print(f"h_n shape:    {h_n.shape}")      
print(f"c_n shape:    {c_n.shape}")  
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm   = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n.squeeze(0)
        out = self.linear(last_hidden)
        return out
model = LSTMClassifier(input_size=1, hidden_size=32, output_size=19)
print(model)

x = torch.randn(8, 2, 1)
print(f"Output shape: {model(x).shape}") 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def generate_dataset(n_samples=10000):
    X = []
    Y = []
    for _ in range(n_samples):
        a = np.random.randint(0, 10)
        b = np.random.randint(0, 10)
        X.append([[a], [b]])     
        Y.append(a + b)            
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int64)

X, Y = generate_dataset(10000)
X = torch.tensor(X)
Y = torch.tensor(Y)

X = X / 9.0


X_train, X_test = X[:8000], X[8000:]
Y_train, Y_test = Y[:8000], Y[8000:]


train_dataset = TensorDataset(X_train, Y_train)
train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)


model     = LSTMClassifier(input_size=1, hidden_size=64, output_size=19)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    epoch_loss = 0
    for X_batch, Y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss   = criterion(output, Y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/50, Loss: {epoch_loss/len(train_loader):.4f}")

with torch.no_grad():
    output = model(X_test)
    pred   = torch.argmax(output, dim=1)
    acc    = (pred == Y_test).float().mean()
    print(f"Test accuracy: {acc:.4f}")

if __name__ == "__main__":

  np.random.seed(0)
  seq_len    = 5
  input_size = 4
  hidden_size = 3

  W_x = np.random.randn(hidden_size, input_size) * 0.1
  W_h = np.random.randn(hidden_size, hidden_size) * 0.1
  b   = np.zeros(hidden_size)

  inputs = np.random.randn(seq_len, input_size)
  h0     = np.zeros(hidden_size)

  hidden_states = rnn_forward(inputs, h0, W_x, W_h, b)
  print(f"hidden_states shape: {hidden_states.shape}")  
  
h = np.array([1.0, 1.0, 1.0])

for t in range(100):
    h = np.tanh(h * 0.99)
print(h)
np.random.seed(0)
input_size  = 4
hidden_size = 3

W_f = np.random.randn(hidden_size, hidden_size + input_size) * 0.1
W_i = np.random.randn(hidden_size, hidden_size + input_size) * 0.1
W_g = np.random.randn(hidden_size, hidden_size + input_size) * 0.1
W_o = np.random.randn(hidden_size, hidden_size + input_size) * 0.1
b_f = np.zeros(hidden_size)
b_i = np.zeros(hidden_size)
b_g = np.zeros(hidden_size)
b_o = np.zeros(hidden_size)

x_t    = np.array([1.0, 0.5, -0.3, 0.8])
h_prev = np.zeros(hidden_size)
c_prev = np.zeros(hidden_size)

h_t, c_t = lstm_step(x_t, h_prev, c_prev, W_f, W_i, W_g, W_o, b_f, b_i, b_g, b_o)
print(f"h_t shape: {h_t.shape}")  
print(f"c_t shape: {c_t.shape}")   
np.random.seed(0)
seq_len     = 5
input_size  = 4
hidden_size = 3

W_f = np.random.randn(hidden_size, hidden_size + input_size) * 0.1
W_i = np.random.randn(hidden_size, hidden_size + input_size) * 0.1
W_g = np.random.randn(hidden_size, hidden_size + input_size) * 0.1
W_o = np.random.randn(hidden_size, hidden_size + input_size) * 0.1
b_f = np.zeros(hidden_size)
b_i = np.zeros(hidden_size)
b_g = np.zeros(hidden_size)
b_o = np.zeros(hidden_size)

inputs = np.random.randn(seq_len, input_size)
h0     = np.zeros(hidden_size)
c0     = np.zeros(hidden_size)

hidden_states = lstm_forward(inputs, h0, c0, W_f, W_i, W_g, W_o, b_f, b_i, b_g, b_o)
print(f"hidden_states shape: {hidden_states.shape}")
print(hidden_states)