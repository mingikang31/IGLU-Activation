import torch 
import torch.nn as nn 
import torch.nn.functional as F

"""
[1] GELU with adjustable parameter a
[2] SiLU with adjustable parameter a
[3] IGLU with adjustable parameter s
"""
class GELU_s(nn.Module):
    def __init__(self, sigma=None, inplace=False):
        super(GELU_s, self).__init__()

        self.sigma = sigma if sigma else nn.Parameter(torch.tensor(5.0))
        self.kAlpha = 0.70710678118654752440

    def forward(self, x):
        return x * 0.5 * (1 + torch.erf(self.sigma * x * self.kAlpha))

class SiLU_s(nn.Module):
    def __init__(self, sigma=None, inplace=False):
        super(SiLU_s, self).__init__()
        
        self.sigma = sigma if sigma else nn.Parameter(torch.tensor(5.0))
        
    def forward(self, x):
        return x * torch.sigmoid(self.sigma * x)

class IGLU_Old(nn.Module):
    def __init__(self, sigma=None, inplace=False):
        super(IGLU_Old, self).__init__()

        self.sigma = sigma if sigma else nn.Parameter(torch.tensor(5.0))
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        return x * (2 * (1/4 + 1/(2 * torch.pi) * torch.arctan(self.sigma * x)))

"""
[1] ArcTan (Gating Function)
[2] ArcTan Approximation (Gating Function)
[3] IGLU (Activation Function) using ArcTan
[4] IGLU Approximation (Activation Function) using ArcTan Approximation
"""
class ArcTan(nn.Module):
    def __init__(self, sigma=None):
        super(ArcTan, self).__init__()

        self.sigma = sigma if sigma else nn.Parameter(torch.tensor(5.0))

    def forward(self, x):
        return 0.5 + (1.0 / torch.pi) * torch.arctan(self.sigma * x)

class ArcTan_Approx(nn.Module):
    def __init__(self, sigma=None):
        super(ArcTan_Approx, self).__init__()
        
        self.sigma = sigma if sigma else nn.Parameter(torch.tensor(5.0))

    def forward(self, x): 
        z = self.sigma * x 
        return (0.5 + torch.clamp(z, min=0)) / (1.0 + torch.abs(z))

class ArcTan_TensorChange(nn.Module):
    def __init__(self, sigma=None):
        super(ArcTan_TensorChange, self).__init__()

        self.sigma = torch.tensor(sigma) if sigma else nn.Parameter(torch.tensor(5.0))

    def forward(self, x):
        return torch.tensor(0.5) + (torch.tensor(1.0) / torch.pi) * torch.arctan(self.sigma * x)

class ArcTan_Approx_TensorChange(nn.Module):
    def __init__(self, sigma=None):
        super(ArcTan_Approx_TensorChange, self).__init__()
        
        self.sigma = torch.tensor(sigma) if sigma else nn.Parameter(torch.tensor(5.0))

    def forward(self, x): 
        z = self.sigma * x 
        return (torch.tensor(0.5) + torch.clamp(z, min=0)) / (torch.tensor(1.0) + torch.abs(z))

class IGLU(nn.Module):
    def __init__(self, sigma=None):
        super(IGLU, self).__init__()
        self.arctan = ArcTan(sigma)
        
    def forward(self, x):
        return x * self.arctan(x)

class IGLU_Approx(nn.Module):
    def __init__(self, sigma=None):
        super(IGLU_Approx, self).__init__()
        self.arctan_approx = ArcTan_Approx(sigma)
        
    def forward(self, x):
        return x * self.arctan_approx(x)

class SquarePlus(nn.Module):
    def __init__(self, beta=4):
        super(SquarePlus, self).__init__()
        self.beta = beta 

    def forward(self, x):
        return 0.5 * (x + torch.sqrt(x**2 + self.beta))


if __name__ == "__main__":
        
    # Activation Selection
    activation = "iglu"
    inplace = True
    sigma = 1.0

    # Activation function mapping
    activation_map = {
        "relu": lambda: nn.ReLU(inplace=inplace), 
        "silu": lambda: nn.SiLU(inplace=inplace), 
        "gelu": lambda: nn.GELU(), 
        "sigmoid": lambda: nn.Sigmoid(), 

        # Previous Activation Generation
        "gelu_s": lambda: GELU_s(sigma=sigma, inplace=inplace), 
        "silu_s": lambda: SiLU_s(sigma=sigma, inplace=inplace), 
        "iglu_old": lambda: IGLU_Old(sigma=sigma, inplace=inplace), 

        # Current Activation Generation 
        "arctan": lambda: ArcTan(sigma=sigma), 
        "arctan_approx": lambda: ArcTan_Approx(sigma=sigma), 
        "iglu": lambda: IGLU(sigma=sigma), 
        "iglu_approx": lambda: IGLU_Approx(sigma=sigma) 
    }

    if activation not in activation_map:
        raise ValueError(f"Unsupported activation function: {activation}")
        
    activation_function = activation_map[activation]()