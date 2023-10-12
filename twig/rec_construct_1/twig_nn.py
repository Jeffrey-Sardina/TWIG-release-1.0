'''
==========================
Neural Network Definitions
==========================
'''
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torcheval.metrics.functional import r2_score
import torch.nn.functional as F

'''
===============
Reproducibility
===============
'''
torch.manual_seed(42)
    
class NeuralNetwork_HPs_v1(nn.Module):
    '''From umls-recommender-nn-TWM-batch_v2'''
    def __init__(self, n):
        super().__init__()
        self.linear1 = nn.Linear(
            in_features=n,
            out_features=10
        )
        self.relu_1 = nn.ReLU()
        # self.dropout_1 = nn.Dropout(p = 0.5)

        self.linear2 = nn.Linear(
            in_features=10,
            out_features=10
        )
        self.relu_2 = nn.ReLU()
        # self.dropout_2 = nn.Dropout(p = 0.5)

        self.linear3 = nn.Linear(
            in_features=10,
            out_features=10
        )
        self.relu_3 = nn.ReLU()
        # self.dropout_3 = nn.Dropout(p = 0.5)

        self.linear_final = nn.Linear(
            in_features=10,
            out_features=1
        )
        self.relu_final = nn.ReLU()

    def forward(self, R_pred):
        #shuffle; https://stackoverflow.com/questions/44738273/torch-how-to-shuffle-a-tensor-by-its-rows
        R_pred = R_pred[torch.randperm(R_pred.size()[0])]

        R_pred = self.linear1(R_pred)
        R_pred = self.relu_1(R_pred)
        # R_pred = self.dropout_1(R_pred)

        R_pred = self.linear2(R_pred)
        R_pred = self.relu_2(R_pred)
        # R_pred = self.dropout_2(R_pred)

        R_pred = self.linear3(R_pred)
        R_pred = self.relu_3(R_pred)
        # R_pred = self.dropout_3(R_pred)

        R_pred = self.linear_final(R_pred)
        R_pred = self.relu_final(R_pred) + 1 #min rank is 1, not 0. Only do this on the last ReLU, I think

        mrr_pred = torch.mean(1 / R_pred)
        return R_pred, mrr_pred    

class NeuralNetwork_HPs_v2(nn.Module):
    '''From umls-recommender-nn-TWM-batch_v3'''
    def __init__(self, n_struct, n_hps):
        super().__init__()
        self.n_struct = n_struct
        self.n_hps = n_hps

        # struct parts are from the version with no hps included
        # we now want to cinclude hps, hwoever
        self.linear_struct_1 = nn.Linear(
            in_features=n_struct,
            out_features=10
        )
        self.relu_1 = nn.ReLU()
        
        self.linear_struct_2 = nn.Linear(
            in_features=10,
            out_features=10
        )
        self.relu_2 = nn.ReLU()

        self.linear_hps_1 = nn.Linear(
            in_features=n_hps,
            out_features=6
        )
        self.relu_3 = nn.ReLU()

        self.linear_integrate_1 = nn.Linear(
            in_features=6 + 10,
            out_features=8
        )
        self.relu_4 = nn.ReLU()

        # self.linear_integrate_2 = nn.Linear(
        #     in_features=8,
        #     out_features=8
        # )
        # self.relu_5 = nn.ReLU()

        self.linear_final = nn.Linear(
            in_features=8,
            out_features=1
        )
        self.relu_final = nn.ReLU()

    def forward(self, X):
        X_struct, X_hps = X[:, :self.n_struct], X[:, self.n_struct:]

        X_struct = self.linear_struct_1(X_struct)
        X_struct = self.relu_1(X_struct)

        X_struct = self.linear_struct_2(X_struct)
        X_struct = self.relu_2(X_struct)

        X_hps = self.linear_hps_1(X_hps)
        X_hps = self.relu_3(X_hps)

        X = self.linear_integrate_1(
            torch.concat(
                [X_struct, X_hps],
                dim=1
            ),
        )
        X = self.relu_4(X)

        # X = self.linear_integrate_2(X)
        # X = self.relu_5(X)

        R_pred = self.linear_final(X)
        R_pred = self.relu_final(R_pred) + 1 #min rank is 1, not 0. Only do this on the last ReLU, I think

        mrr_pred = (1 / R_pred).mean() #R_pred.round()
        return R_pred, mrr_pred
