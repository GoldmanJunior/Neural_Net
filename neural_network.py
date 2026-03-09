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