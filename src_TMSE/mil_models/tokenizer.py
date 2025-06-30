import torch
import torch.nn as nn

class PrototypeTokenizer(nn.Module):
    """
    Tokenize the prototype features, so that we have access to mixture probabilities/mean/covariance
    """
    def __init__(self, proto_model_type='Gaussian', out_type='param_cat', p=8):
        super().__init__()
        self.model_type = proto_model_type
        self.p = p
        self.out_type = out_type

    def get_eff_dim(self):
        return 2 * self.p

    def forward(self, X):
        n_samples = X.shape[0]

        if self.model_type == 'Gaussian':
            if self.out_type == 'allcat' or self.out_type == 'weight_param_cat':
                d = (X.shape[1] - self.p) // (2 * self.p)
                prob = X[:, : self.p]
                mean = X[:, self.p: self.p * (1 + d)].reshape(-1, self.p, d)
                cov = X[:, self.p * (1 + d):].reshape(-1, self.p, d)
            else:
                raise NotImplementedError(f"Not implemented for {self.out_type}")
        else:
            raise NotImplementedError(f"Not implemented for {self.model_type}")


        return prob, mean, cov
