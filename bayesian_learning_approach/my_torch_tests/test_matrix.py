import numpy as np
import torch
import torch.nn as nn

from utils import matprint


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # nn.Linear(2, 16),
            # # nn.ReLU(),
            # nn.Linear(16, 16),
            # # nn.ReLU(),
            # nn.Linear(16, 2),
            nn.Linear(2, 2),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class CustomArrayDataset(torch.utils.data.Dataset):
    def __init__(self, N_obs=1000):
        dim = 2
        self.real_matrix = np.array([[1.0,12.0],[0.0,1.0]])
        self.parameters = np.random.default_rng().normal(4, 6, (dim, N_obs))
        self.observables = np.array(self.real_matrix @ self.parameters) + np.random.default_rng().normal(1, 1, (dim, N_obs))
        self.parameters = self.parameters.T
        self.observables = self.observables.T

    def __len__(self):
        return len(self.parameters)

    def __getitem__(self, idx):
        return self.observables[idx], self.parameters[idx]


if __name__ == '__main__':

    # We create the dataset

    # dim = 2
    # N_obs = 100

    # real_matrix = np.array([[1.0,2.0],[0.0,1.0]])

    # matprint(real_matrix)
    # parameters = np.round(np.random.default_rng().normal(4, 6, (dim, N_obs)))
    # observables = np.array(real_matrix @ parameters)

    # matprint(parameters)
    # print("and")
    # matprint(observables)

    # tensors_obs = []
    # tensors_labels = []
    # for label, obs in zip(parameters.T, observables.T):
    #     tensors_labels.append(torch.tensor( label))
    #     tensors_obs.append(torch.tensor(obs))

    # We set the dataset

    # dataset = torch.utils.data.TensorDataset(np.array(tensors_obs), np.array(tensors_labels))
    dataset = CustomArrayDataset()
    # print(dataset.targets)

    loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    # We create the NN

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    test_dataset = CustomArrayDataset(100)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)


    model = NeuralNetwork().to(device)

    # We check the model

    random_entry = torch.rand(1,2,1,device=device)
    logits = model(random_entry)
    print(random_entry)
    print(f"Predicted class: {logits}")


    # We select these hyperparameters

    learning_rate = 0.1
    batch_size = len(dataset)
    epochs = 10


    # We select the loss function

    loss_fn = nn.MSELoss()

    # We select this optimizer

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # We train
    size = len(loader.dataset)
    for batch, values in enumerate(loader):
        X = values[0]
        y = values[1]
        # print(X)
        # print(y)
        # Compute prediction and loss
        pred = model(y.float())
        # print(f"X: {X}")
        # print(f"pred: {pred}")
        # print(f"y: {y}")
        loss = loss_fn(pred, X.float())
        print(f"loss: {loss:>3f}")

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(batch)
        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            # print(f"loss: {loss:>3f}", end="\r")
        # print("\r", ("\tEpoch {}/{}: Train step {:04d}/{} prob = {:.4f} model = {:.4f} loss = {:.4f}          ").format(
        # 																								n+1, args.nepochs,
        # 																								batch_id+1,
        # 																								batchLen,
        # 																								torch.exp(-logprob.detach().cpu()).item(),
        # 																								modelloss.detach().cpu().item(),
        # 																								l.detach().cpu().item()), end="")

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            print(param.data)

    error = 0
    with torch.no_grad():
        for values in test_loader:
            X = values[0]
            y = values[1]
            # print(X)
            # print(y)
            # Compute prediction and loss
            pred = model(y.float())
            # print(f"X: {X}")
            # print(f"pred: {pred}")
            # print(f"y: {y}")
            diff = pred - X
            error += diff[0,0]**2 + diff[0,1]**2
    mse = error / len(test_loader)
    print(f"MSE on test is {mse}")
