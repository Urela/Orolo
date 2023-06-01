import optim
from tensor import Tensor
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from MNIST_DATASET import fetch_mnist

# np.random.seed(9) # simulation 9, kekw
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
optim = optim.SGD([model.fc1, model.fc2], lr=0.001)

# ********************************************************************

def train(model, optim, steps, batch_size, losses:list, accuracies:list):
    for epoch in tqdm(range(steps)):
        samp = np.random.randint(0, x_train.shape[0], size=(batch_size))
        x, y = Tensor(x_train[samp].reshape((-1, 28*28))), Tensor(y_train[samp].reshape(-1,10))

        out = model.forward(x)
        loss = out.mul(y).mean() # NLL loss function

        optim.zero_grad()
        loss.backward()
        optim.step()

        losses.append(loss.data)

        pred = np.argmax(out.data, axis=1)
        targ = np.argmax(y.data, axis=1)
        accuracies.append( (pred == targ).mean() )
    return losses, accuracies

losses, accuracies = train(model, optim, steps=1000, batch_size=128, losses=[], accuracies=[])
# *************************** Evaluation ***************************  
outputs = model.forward(Tensor(x_test.reshape(-1, 28*28)))
outputs = np.argmax(outputs.numpy(), axis=1)
accuracy = (y_test == outputs).mean()
print(f"model accuracy {accuracy:.3f}")

# *************************** Metrics ***************************  
fig, ax = plt.subplots(1,1)
fig.suptitle('MNIST performance')
ax.plot(np.arange(len(losses)), losses)
ax.set( xlabel='Iteration', ylabel='losses',title='Losses')
fig.savefig("test.png")
plt.show()  


