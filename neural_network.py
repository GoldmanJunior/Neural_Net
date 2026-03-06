import numpy as np

def sigmoid(z):
  output= 1/(1+np.exp(-z))
  return output

def relu(z):
  output = np.maximum(0,z)
  return output

def mse_loss(y_true,y_pred):
  y_pred=np.array(y_pred)
  y_true=np.array(y_true)
  n=len(y_true)
  loss=(1/n)*np.sum((y_pred-y_true)**2)
  return loss

def dense_layer(x,w ,b):
  x = np.array(x)
  w = np.array(w)
  z= np.dot(x,w.T)+b
  return z

def forward(x,w1,b1,w2,b2):
  d1=dense_layer(x,w1,b1)
  a1=relu(d1)
  d2=dense_layer(a1,w2,b2)
  a2=sigmoid(d2)
  return d1,a1,d2,a2

def backward(x,w1,b1,w2,b2,a1,a2,z1,z2,y_true):
  dL_da2=2 *(a2-y_true)
  sigmoid_prime_z2 = a2 * (1 - a2)
  delta2=dL_da2 * sigmoid_prime_z2
  dW2 = np.dot(delta2.reshape(-1, 1), a1.reshape(1, -1))
  db2=delta2
  delta1=np.dot(delta2, w2) * (z1 > 0)
  dw1=np.dot(delta1.reshape(-1, 1), x.reshape(1, -1))
  db1=delta1
  return dw1, db1, dW2, db2

def train(x,y_true,w1,b1,w2,b2,learning_rate,epochs):
  for epoch in range(epochs):
    z1,a1,z2,a2=forward(x,w1,b1,w2,b2)
    loss = mse_loss(y_true, a2)
    dw1,db1,dw2,db2=backward(x,w1,b1,w2,b2,a1,a2,z1,z2,y_true)
    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2
    if epoch % 100 == 0:
      print(f"Epoch: {epoch}, Loss: {loss}")

if __name__ == "__main__":
  #test the mse_loss function
  print("Testing mse_loss function:")
  y_pred = np.array([0.9, 0.2, 0.8])
  y_true = np.array([1.0, 0.0, 1.0])
  loss= mse_loss(y_true, y_pred)
  print(loss)  
  #test the dense_layer function
  print("Testing dense_layer function:")
  x = np.array([1.0, 2.0])
  W = np.array([[0.5, -0.2],
                [0.1,  0.8],
                [0.3,  0.4]])
  b = np.array([0.1, 0.2, 0.3])
  print(dense_layer(x, W, b))
  #test the forward function
  print("Testing forward function:")
  np.random.seed(0)
  x  = np.array([1.0, 2.0, 3.0])
  W1 = np.random.randn(4, 3)   
  b1 = np.zeros(4)
  print(f"voici b1: {b1}")
  W2 = np.random.randn(1, 4) 
  print(W2)
  b2 = np.zeros(1)
  print(forward(x, W1, b1, W2, b2))
  print("Testing backward function:")
  z1, a1, z2, a2 = forward(x, W1, b1, W2, b2)
  print(backward(x, W1, b1, W2, b2, a1, a2, z1, z2, y_true=np.array([1.0])))
  #test the train function
  print("Testing train function:")
  np.random.seed(0)
  x      = np.array([1.0, 2.0, 3.0])
  y_true = np.array([0.0])
  W1     = np.random.randn(4, 3)
  b1     = np.zeros(4)
  W2     = np.random.randn(1, 4)
  b2     = np.zeros(1)

  train(x, y_true, W1, b1, W2, b2, learning_rate=0.01, epochs=1000)