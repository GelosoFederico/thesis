from random import choice, random
import torch
import torch.nn as nn
from test_matrix import CustomArrayDataset, NeuralNetwork

if __name__ == "__main__":

    dataset = CustomArrayDataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_dataset = CustomArrayDataset(1000)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    N_models = 6
    learning_rate = 0.1
    batch_size = len(dataset)
    epochs = 10
    size = len(loader.dataset)
    models = []

    for i in range(N_models):
        model = NeuralNetwork().to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for batch, values in enumerate(loader):
            X = values[0]
            y = values[1]
            # Compute prediction and loss
            pred = model(y.float())
            # print(f"X: {X}")
            # print(f"pred: {pred}")
            # print(f"y: {y}")
            loss = loss_fn(pred, X.float())
            # print(f"loss: {loss:>3f}")

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(batch)
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
        print(f"Final loss: {loss}")
        models.append(model)

    error = 0
    with torch.no_grad():
        for values in test_loader:
            X = values[0]
            y = values[1]
            # print(X)
            # print(y)
            # Compute prediction and loss
            model = choice(models)
            pred = model(y.float())
            # print(f"X: {X}")
            # print(f"pred: {pred}")
            # print(f"y: {y}")
            diff = pred - X
            error += diff[0,0]**2 + diff[0,1]**2
    mse = error / len(test_loader)
    print(f"MSE on test with all models is {mse}")

    error = 0
    model = models[0]
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
    print(f"MSE on test with first model is {mse}")
