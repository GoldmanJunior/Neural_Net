import numpy as np
from urllib import request
import gzip
import os

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

def sigmoid(z):
  output= 1/(1+np.exp(-z))
  return output

def relu(z):
  output = np.maximum(0,z)
  return output

def softmax(z):
  z=np.array(z)
  z = z - np.max(z)
  return np.exp(z)/np.sum(np.exp(z))

def cross_entropy_loss(y_true,y_pred):
  y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
  loss=-np.sum(y_true * np.log(y_pred))
  return loss

def one_hot(labels, n_classes=10):
    n = len(labels)
    one_hot_labels = np.zeros((n, n_classes))
    one_hot_labels[np.arange(n), labels] = 1
    return one_hot_labels

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
  a2=relu(d2)
  return d1,a1,d2,a2

def forward1(x,w1,b1,w2,b2,w3,b3):
  d1=dense_layer(x,w1,b1)
  a1=relu(d1)
  d2=dense_layer(a1,w2,b2)
  a2=relu(d2)
  d3=dense_layer(a2,w3,b3)
  a3=softmax(d3)
  return d1,a1,d2,a2,d3,a3

def backward(x, y_true, W1, b1, W2, b2, W3, b3, a1, a2, a3, z1, z2, z3):
    delta3 = a3 - y_true
    dW3 = np.dot(delta3.reshape(-1, 1), a2.reshape(1, -1))
    db3 = delta3
    delta2 = np.dot(delta3, W3) * (z2 > 0)
    dW2 = np.dot(delta2.reshape(-1, 1), a1.reshape(1, -1))
    db2 = delta2
    delta1 = np.dot(delta2, W2) * (z1 > 0)
    dW1 = np.dot(delta1.reshape(-1, 1), x.reshape(1, -1))
    db1 = delta1

    return dW1, db1, dW2, db2, dW3, db3

def train(x,y_true,w1,b1,w2,b2,learning_rate,epochs):
  for epoch in range(epochs):
    z1,a1,z2,a2=forward(x,w1,b1,w2,b2)
    loss = mse_loss(y_true, a2)
    dw1,db1,dw2,db2,dw3,db3=backward(x,w1,b1,w2,b2,w3,b3,a1,a2,z1,z2,z3)
    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2
    w3 -= learning_rate * dw3
    b3 -= learning_rate * db3
    if epoch % 100 == 0:
      print(f"Epoch: {epoch}, Loss: {loss}")

def train_mnist(train_images, train_labels_oh, W1, b1, W2, b2,W3, b3, 
                learning_rate=0.01, epochs=10, batch_size=32):
    
    n = len(train_images)
    t=0

    m_W1 = np.zeros_like(W1)
    v_W1 = np.zeros_like(W1)
    m_b1 = np.zeros_like(b1)
    v_b1 = np.zeros_like(b1)
    m_W2 = np.zeros_like(W2)
    v_W2 = np.zeros_like(W2)
    m_b2 = np.zeros_like(b2)
    v_b2 = np.zeros_like(b2)
    m_W3 = np.zeros_like(W3)
    v_W3 = np.zeros_like(W3)
    m_b3 = np.zeros_like(b3)
    v_b3 = np.zeros_like(b3)
    
    for epoch in range(epochs):

        indices = np.random.permutation(n)
        X_shuffled = train_images[indices]
        Y_shuffled = train_labels_oh[indices]
        
        epoch_loss = 0
        
        for i in range(0, n, batch_size):

            X_batch = X_shuffled[i:i+batch_size]
            Y_batch = Y_shuffled[i:i+batch_size]
            
            dW1_total = np.zeros_like(W1)
            db1_total = np.zeros_like(b1)
            dW2_total = np.zeros_like(W2)
            db2_total = np.zeros_like(b2)
            dW3_total = np.zeros_like(W3)
            db3_total = np.zeros_like(b3)


            for j in range(len(X_batch)):
                x = X_batch[j]
                y = Y_batch[j]
                
                z1, a1, z2, a2, z3, a3 = forward1(x, W1, b1, W2, b2, W3, b3)
                
                epoch_loss += cross_entropy_loss(y, a3)
                
                dw1, db1_grad, dw2, db2_grad, dw3, db3_grad = backward(x, y, W1, b1, W2, b2, W3, b3, a1, a2, a3, z1, z2, z3)
                
                dW1_total += dw1
                db1_total += db1_grad
                dW2_total += dw2
                db2_total += db2_grad
                dW3_total += dw3
                db3_total += db3_grad

            dW1_total /= len(X_batch)
            db1_total /= len(X_batch)
            dW2_total /= len(X_batch)
            db2_total /= len(X_batch)
            dW3_total /= len(X_batch)
            db3_total /= len(X_batch)

            t += 1

            m_W1= 0.9 * m_W1 + 0.1 * dW1_total
            v_W1= 0.999 * v_W1 + 0.001 * (dW1_total ** 2)
            m_W1_corr = m_W1 / (1 - 0.9**t)
            v_W1_corr = v_W1 / (1 - 0.999**t)
            m_b1= 0.9 * m_b1 + 0.1 * db1_total
            v_b1= 0.999 * v_b1 + 0.001 * (db1_total ** 2)
            m_b1_corr = m_b1 / (1 - 0.9**t)
            v_b1_corr = v_b1 / (1 - 0.999**t)
            m_W2= 0.9 * m_W2 + 0.1 * dW2_total
            v_W2= 0.999 * v_W2 + 0.001 * (dW2_total ** 2)
            m_W2_corr = m_W2 / (1 - 0.9**t)
            v_W2_corr = v_W2 / (1 - 0.999**t)
            m_b2= 0.9 * m_b2 + 0.1 *  db2_total
            v_b2= 0.999 * v_b2 + 0.001 * (db2_total ** 2)
            m_b2_corr = m_b2 / (1 - 0.9**t)
            v_b2_corr = v_b2 / (1 - 0.999**t)
            m_W3= 0.9 * m_W3 + 0.1 * dW3_total
            v_W3= 0.999 * v_W3 + 0.001 * (dW3_total ** 2)
            m_W3_corr = m_W3 / (1 - 0.9**t)
            v_W3_corr = v_W3 / (1 - 0.999**t)
            m_b3= 0.9 * m_b3 + 0.1 * db3_total
            v_b3= 0.999 * v_b3 + 0.001 * (db3_total ** 2)
            m_b3_corr = m_b3 / (1 - 0.9**t)
            v_b3_corr = v_b3 / (1 - 0.999**t)

            W1 -= learning_rate * m_W1_corr / (np.sqrt(v_W1_corr) + 1e-8)
            b1 -= learning_rate * m_b1_corr / (np.sqrt(v_b1_corr) + 1e-8)
            W2 -= learning_rate * m_W2_corr / (np.sqrt(v_W2_corr) + 1e-8)
            b2 -= learning_rate * m_b2_corr / (np.sqrt(v_b2_corr) + 1e-8)
            W3 -= learning_rate * m_W3_corr / (np.sqrt(v_W3_corr) + 1e-8)
            b3 -= learning_rate * m_b3_corr / (np.sqrt(v_b3_corr) + 1e-8)

        epoch_loss /= n
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    
    return W1, b1, W2, b2, W3, b3

def accuracy(images, labels, W1, b1, W2, b2, W3, b3):
    correct = 0
    for i in range(len(images)):
        _, _, _, _, _, a3 = forward1(images[i], W1, b1, W2, b2, W3, b3)
        pred = np.argmax(a3)     
        true = np.argmax(labels[i])  
        if pred == true:
            correct += 1
    return correct / len(images)

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
  #print("Testing backward function:")
  #z1, a1, z2, a2 = forward(x, W1, b1, W2, b2)
  #print(backward(x, W1, b1, W2, b2, a1, a2, z1, z2, y_true=np.array([1.0])))
  #test the train function
  print("Testing train function:")
  np.random.seed(0)
  x      = np.array([1.0, 2.0, 3.0])
  y_true = np.array([0.0])
  W1     = np.random.randn(4, 3)
  b1     = np.zeros(4)
  W2     = np.random.randn(1, 4)
  b2     = np.zeros(1)

  #train(x, y_true, W1, b1, W2, b2, learning_rate=0.01, epochs=1000)
  #softmax test
  print("Testing softmax function:")
  z = np.array([2.0, 1.0, 0.1])
  print(softmax(z))  
  print(np.sum(softmax(z)))
  print(softmax(np.array([2.0, 1.0, 0.1]))) 
  print(softmax(np.array([1000.0, 2000.0, 3000.0])))
  #cross_entropy_loss test
  print("Testing cross_entropy_loss function:")
  y_true = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
  y_pred = np.array([0.01, 0.01, 0.9, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02])
  print(cross_entropy_loss(y_true, y_pred)) 
#test the load_mnist 
  print("Testing load_mnist:")
  train_images, train_labels, test_images, test_labels = load_mnist()
  print(f"Train images shape: {train_images.shape}")
  print(f"Train labels shape: {train_labels.shape}")
  print(f"Test images shape:  {test_images.shape}")
  print(f"Test labels shape:  {test_labels.shape}")

train_images = train_images / 255.0
test_images  = test_images / 255.0

train_labels_oh = one_hot(train_labels)
test_labels_oh  = one_hot(test_labels)

print(train_images.max())       
print(train_images.min())       
print(train_labels_oh[0])       
print(train_labels_oh.shape)    
# Initialize weights and biases 
np.random.seed(42)
W1 = np.random.randn(128, 784) * 0.01
b1 = np.zeros(128)
W2 = np.random.randn(10, 128) * 0.01
b2 = np.zeros(10)
# text the forward1 function
print("Testing forward1 function:")
np.random.seed(42)
W1 = np.random.randn(128, 784) * 0.01
b1 = np.zeros(128)
W2 = np.random.randn(64, 128) * 0.01
b2 = np.zeros(64)
W3 = np.random.randn(10, 64) * 0.01
b3 = np.zeros(10)

x = train_images[0]
z1, a1, z2, a2, z3, a3 = forward1(x, W1, b1, W2, b2, W3, b3)
print(f"a3 shape: {a3.shape}")       
print(f"a3 sum: {np.sum(a3):.4f}")   

# Test one image
x = train_images[0]
z1, a1, z2, a2, z3, a3 = forward1(x, W1, b1, W2, b2, W3, b3)
print(f"a3 shape: {a3.shape}")       
print(f"a3 sum: {np.sum(a3):.4f}")  
print(f"a3: {a3}")

print("Training on MNIST:")
np.random.seed(42)
W1 = np.random.randn(128, 784) * np.sqrt(2.0 / 784)
b1 = np.zeros(128)
W2 = np.random.randn(64, 128) * np.sqrt(2.0 / 128)
b2 = np.zeros(64)
W3 = np.random.randn(10, 64) * np.sqrt(2.0 / 64)
b3 = np.zeros(10)

W1, b1, W2, b2, W3, b3 = train_mnist(
    train_images, train_labels_oh,
    W1, b1, W2, b2, W3, b3,
    learning_rate=0.001,
    epochs=20,
    batch_size=32
)

print("Evaluating on test set:")
train_acc = accuracy(train_images, train_labels_oh, W1, b1, W2, b2, W3, b3)
test_acc  = accuracy(test_images, test_labels_oh, W1, b1, W2, b2, W3, b3)
print(f"Train accuracy: {train_acc:.4f}")
print(f"Test accuracy:  {test_acc:.4f}")
