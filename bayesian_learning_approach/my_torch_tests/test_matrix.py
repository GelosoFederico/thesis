import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal
from utils import matprint

class VIModule(nn.Module) :
    """
    A mixin class to attach loss functions to layer. This is usefull when doing variational inference with deep learning.
    """
    
    def __init__(self, *args, **kwargs) :
        super().__init__(*args, **kwargs)
        
        self._internalLosses = []
        self.lossScaleFactor = 1
        
    def addLoss(self, func) :
        self._internalLosses.append(func)
        
    def evalLosses(self) :
        t_loss = 0
        
        for l in self._internalLosses :
            t_loss = t_loss + l(self)
            
        return t_loss
    
    def evalAllLosses(self) :
        
        t_loss = self.evalLosses()*self.lossScaleFactor
        
        for m in self.children() :
            if isinstance(m, VIModule) :
                t_loss = t_loss + m.evalAllLosses()*self.lossScaleFactor
                
        return t_loss


class MeanFieldGaussianFeedForward(VIModule):
    """
    A feed forward layer with a Gaussian prior distribution and a Gaussian variational posterior.
    """

    def __init__(self,
            in_features,
            out_features,
            bias = True,
            groups=1,
            weightPriorMean = 0,
            weightPriorSigma = 1.,
            biasPriorMean = 0,
            biasPriorSigma = 1.,
            initMeanZero = False,
            initBiasMeanZero = False,
            initPriorSigmaScale = 0.01) :


        super(MeanFieldGaussianFeedForward, self).__init__()

        self.samples = {'weights' : None, 'bias' : None, 'wNoiseState' : None, 'bNoiseState' : None}

        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        self.weights_mean = Parameter((0. if initMeanZero else 1.)*(torch.rand(out_features, int(in_features/groups))-0.5))
        self.lweights_sigma = Parameter(torch.log(initPriorSigmaScale*weightPriorSigma*torch.ones(out_features, int(in_features/groups))))
 
        self.noiseSourceWeights = Normal(torch.zeros(out_features, int(in_features/groups)), 
                                torch.ones(out_features, int(in_features/groups)))

        self.addLoss(lambda s : 0.5*s.getSampledWeights().pow(2).sum()/weightPriorSigma**2)
        self.addLoss(lambda s : -self.out_features/2*np.log(2*np.pi) - 0.5*s.samples['wNoiseState'].pow(2).sum() - s.lweights_sigma.sum())

        if self.has_bias :
            self.bias_mean = Parameter((0. if initBiasMeanZero else 1.)*(torch.rand(out_features)-0.5))
            self.lbias_sigma = Parameter(torch.log(initPriorSigmaScale*biasPriorSigma*torch.ones(out_features)))

            self.noiseSourceBias = Normal(torch.zeros(out_features), torch.ones(out_features))

            self.addLoss(lambda s : 0.5*s.getSampledBias().pow(2).sum()/biasPriorSigma**2)
            self.addLoss(lambda s : -self.out_features/2*np.log(2*np.pi) - 0.5*s.samples['bNoiseState'].pow(2).sum() - self.lbias_sigma.sum())


    def sampleTransform(self, stochastic=True) :
        self.samples['wNoiseState'] = self.noiseSourceWeights.sample().to(device=self.weights_mean.device)
        self.samples['weights'] = self.weights_mean + (torch.exp(self.lweights_sigma)*self.samples['wNoiseState'] if stochastic else 0)
        
        if self.has_bias :
            self.samples['bNoiseState'] = self.noiseSourceBias.sample().to(device=self.bias_mean.device)
            self.samples['bias'] = self.bias_mean + (torch.exp(self.lbias_sigma)*self.samples['bNoiseState'] if stochastic else 0)
        
    def getSampledWeights(self) :
        return self.samples['weights']
    
    def getSampledBias(self) :
        return self.samples['bias']
    
    def forward(self, x, stochastic=True) :
        
        self.sampleTransform(stochastic=stochastic)
        
        return nn.functional.linear(x, self.samples['weights'], bias = self.samples['bias'] if self.has_bias else None)


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
        )
        self.linear1 = MeanFieldGaussianFeedForward(2, 2,
                                            weightPriorSigma = 1,
                                            biasPriorSigma = 5,
                                            initPriorSigmaScale=1e-7)
        self.lossScaleFactor = 1
        self._internalLosses = []

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear1(x, stochastic=True)
        return logits

    def evalLosses(self) :
        t_loss = 0
        
        for l in self._internalLosses :
            t_loss = t_loss + l(self)
            
        return t_loss
    
    def evalAllLosses(self) :
        
        t_loss = self.evalLosses()*self.lossScaleFactor
        
        for m in self.children() :
            if isinstance(m, VIModule) :
                t_loss = t_loss + m.evalAllLosses()*self.lossScaleFactor
                
        return t_loss


class CustomArrayDataset(torch.utils.data.Dataset):
    def __init__(self, N_obs=10000, matrix=None):
        dim = 2
        self.real_matrix = np.array([[1.0,12.0],[0.0,1.0]])
        self.parameters = np.random.default_rng().normal(4, 6, (dim, N_obs))
        noise = np.random.default_rng().normal(0, 1, (dim, N_obs))
        self.observables = np.array(self.real_matrix @ self.parameters) + noise
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
        logprob = loss_fn(pred, X.float())
        print(f"loss: {loss:>3f}")

        # Backpropagation
        optimizer.zero_grad()
        l = len(loader) *logprob

        modelloss = model.evalAllLosses()
        l += modelloss
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
