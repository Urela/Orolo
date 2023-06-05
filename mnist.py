import optim
from tensor import Tensor
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from MNIST_DATASET import fetch_mnist

x_train, y_train, x_test, y_test = fetch_mnist()

# ********************************************************************
class Model:
  def __init__(self):
    self.fc1 = Tensor.uniform(784, 128)
    self.fc2 = Tensor.uniform(128, 10)
  def forward(self, x):
    x = x.matmul(self.fc1).relu()
    x = x.matmul(self.fc2).logsoftmax()
    return x

model = Model()
optim = optim.Adam([model.fc1, model.fc2], lr=0.001)
# ********************************************************************

# sparase categorical cross entropy
#def loss_fxn(pred, targ):

def loss_fxn(out, Y):
  num_classes = out.shape[-1]
  YY = Y.flatten()
  y = np.zeros((YY.shape[0], num_classes), np.float32)
  # correct loss for NLL, torch NLL loss returns one per row
  y[range(y.shape[0]),YY] = -1.0*num_classes
  y = y.reshape(list(Y.shape)+[num_classes])
  y = Tensor(y)
  return out.mul(y).mean()


# ********************************************************************
def train(model, optim, steps, batch_size, losses:list, accuracies:list):
  for epoch in tqdm(range(steps)):
    samp = np.random.randint(0, x_train.shape[0], size=(batch_size))
    #x, y = Tensor(x_train[samp].reshape((-1, 28*28)), requires_grad=False), y_train[samp] #Tensor(y_train[samp])
    x, y = Tensor(x_train[samp], requires_grad=False), y_train[samp] 

    out = model.forward(x)
    #out = Tensor(out.compute())
    #print(x.shape,out.shape, y.shape)

    loss = loss_fxn(out , y) # NLL loss function
    #loss.compute()

    optim.zero_grad()
    loss.backward()
    optim.step()

    losses.append(loss.data)

    pred = np.argmax(out.numpy(), axis=1)
    #print(pred, y)
    accuracies.append( (pred == y).mean() )
  return losses, accuracies
losses, accuracies = train(model, optim, steps=1000, batch_size=128, losses=[], accuracies=[])

# *************************** Evaluation ***************************  
outputs = model.forward(Tensor(x_test.reshape(-1, 28*28)))
outputs = np.argmax(outputs.numpy(), axis=1)
accuracy = (y_test == outputs).mean()
print(f"model accuracy {accuracy:.3f}")

# *************************** Plot Metrics ***************************  
#fig, ax = plt.subplots()
#ax.set_title('MNIST performance')
#ax.plot(np.arange(len(losses)), losses)
#fig.savefig("test.png")
#plt.show()  

