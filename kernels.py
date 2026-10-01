# Matern covariance(s) using sympy and numpy lambdify [via Adrian from the CIROQUO training at Toulouse]
import torch
import functools
from torch import Tensor, nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from typing import Literal
import argparse
import time 
# from scipy import stats
# from problems.burgers1D import Burgers1DRegressor

# import gpytorch as gpt
# from SimpleKrigGpytorch import MyLeaveOneOutPseudoLikelihood, ExactGPModel
# from SimpleKrigGpytorch import fit_regressor as gpy_fit_regressor

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

#===============
# Colored text
#===============
RED, END = '\033[91m', '\033[0m'
printRed = lambda sTxt: print(RED + sTxt + END)

_EPS = 1e-8

def dim_check(x: Tensor, z: Tensor, 
              index: int = -1, 
              **kwargs):
    message: str | None = None
    FLAG: bool = True
    if 'message' in kwargs:
        message = kwargs.get('message')

    try:
        assert x.size(dim=index) == z.size(dim=index)     # Simple AssertionError check
    except AssertionError:
        FLAG = False

    if not FLAG:
        if message == None:
            raise AssertionError(RED+f'\n Inputs are not of the same dimension: x_d = {x.size(dim=index)} != z_d = {z.size(dim=index)} \n'+END)
        raise AssertionError(RED+ '\n' + message + '\n' +END) 

def symmetry_check(x: Tensor, rtol: float | None = None, 
                   atol: float | None = None, matrix: str | None = None) -> Tensor:
    """ Check if given matrix is symmetric """

    args = []
    if not rtol == None:
        args.append(rtol)
    if not atol == None:
        args.append(atol)

    # try:
    #     assert torch.allclose(x, x.T, *args)
    # except AssertionError:
    #     if matrix == None:
    #         print(f'Maximum error: {torch.max(torch.abs(x - x.T))}')
    #         # print(f'Asymmetric matrix: \n {x=} \n and \n {x.T=}')
    #     else: 
    #         print(f'Maximum error in {matrix}: {torch.max(torch.abs(x - x.T))}')
    #         # print(f'Asymmetric matrix: \n {matrix} = {x} \n and \n {matrix}.T = {x.T}')
    #     print(f'Maximum error: {torch.max(torch.abs(x - x.T))}')

    #     # [OPTIONAL] to visualize matrix
    #     # plt.imshow(x.detach().numpy())
    #     # plt.colorbar()
    #     # plt.show()

    #     print('Returning symmetric correction ...')
    #     return (x + x.T) / 2                                    # Optional, completely.

def time_execution(func):
    """A decorator that times a class method if the instance's 
    boolean flag (time_enabled) is set to True.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Check if the boolean flag exists and is True
        if getattr(self, 'time_enabled', False):
            start_time = time.perf_counter()
            result = func(self, *args, **kwargs)
            end_time = time.perf_counter()
            
            elapsed_time = end_time - start_time
            print(f"[Timing] Method '{func.__name__}' took {elapsed_time:.6f} seconds.")
            return result
        else:
            # Run normally without timing if the flag is False
            return func(self, *args, **kwargs)
            
    return wrapper

def _inverse_softplus(value: Tensor) -> Tensor:
    """Return r such that softplus(r) is approximately value."""

    value = torch.max(value, _EPS*torch.ones_like(value))
    return torch.log(torch.expm1(value))

def weighted_squared_distance(x: Tensor, z: Tensor | None = None, 
                              weights: Tensor | None = None,
                                diag: bool = False) -> Tensor:
    """Pairwise-squared Euclidean distances."""

    DIM = x.size(dim=-1)                                # Input dimension

    weights = torch.ones(DIM) if weights is None else weights
    z = x if z is None else z

    # Simple sanity dimension check.
    dim_check(x, z)
    # Sanity check for lenthscale
    dim_check(weights, x, 
              message = f'Invalid lengthscale dimension (DIM ={weights.size(dim=-1)}) for design space.')

    xl = x / weights
    zl = z / weights
    
    x2 = (xl * xl).sum(dim=-1, keepdim=True)      # Column vector
    z2 = (zl * zl).sum(dim=-1).unsqueeze(0)       # Row vector
    if diag:

        # Sanity check: n=m?
        dim_check(xl, zl, index=0, 
                  message = f'diag mode requires n=m but {xl.size(0)} != {zl.size(0)}')
        
        return torch.diag((x2 + z2 - 2.0 * xl @ zl.T).clamp_min(0.0))              # Diagonal mode

    return (x2 + z2 - 2.0 * xl @ zl.T).clamp_min(0.0)

#================================
# Expanded squared distance function.
#================================

def expanded_weighted_squared_distance(x: Tensor, z: Tensor, 
                              weights: Tensor | None = None,
                                diag: bool = False) -> Tensor:
    """`weighted_squared_distance()` of two equisized matrices `x` and `z` inputs.
    
    Usually takes as inputs the output of `expanding_inputs()`
    """

    DIM = x.size(dim=-1)                                # Input dimension

    weights = torch.ones(DIM) if weights is None else weights

    # z = x if z is None else z

    if not x.size() == z.size():
       print(x.size(), z.size())
       raise('x and z must be tensors of same size()') 

    # Sanity check for lenthscale
    dim_check(weights, x, 
              message = f'Invalid lengthscale dimension (DIM ={weights.size(dim=-1)}) for design space.')
    
    r = (x - z) / weights                                       # lengthscale. 
    d = (r * r).sum(dim=-1).clamp_min(0.0)                      # .sum(dim=-1) reduces that dimension, in this case, producing scalars.
    return d

def expanding_inputs(x: Tensor, z:  Tensor | None = None,
                     grad: bool = False) -> Tensor:
    """ Takes two vector-like tensors x=(N,d) & z=(M,d) and produces the cartesian product matrix-like tensors (M X N, d) and (M X N, d) """
    FLAG = True if z is None else False                                             # FLAG to control return behaviour
    z = x if z is None else z

    DIM = x.size(dim=-1)                                                            # Input dimension, DIM = d

    # Simple sanity dimension check.
    dim_check(x, z)                                                                   

    x_clean = x.clone().detach()                                                    # (N,d)
    z_clean = z.clone().detach()                                                    # (M,d)

    x_expanded = x_clean.unsqueeze(dim=0).expand(z.size(dim=0), x.size(dim=0), 
                                DIM).requires_grad_(grad)                           # (M x N,d)

    if FLAG == True:                                                                # When z is None, return only one matrix.
        return x_expanded

    z_expanded = z_clean.unsqueeze(dim=1).expand(z.size(dim=0), x.size(dim=0),      
                                DIM).requires_grad_(grad)                           # (M x N,d)

    return x_expanded, z_expanded


#================================
# Differentiation index retrieval 
#================================

# index retrieval in 1D: `x` or `z` differentiation
def index_arg_1D(indices: list = [1,0],                     # index of differentiation. 
                arg: int = 0):                              # First argument first.

    if not len(indices) == 2:
        raise('Wrong index dimensions for 1D. Please provide a list of length 2.')
    
    if arg == 0:
        x_index = [*(indices[0] * [1]), *(indices[1] * [0])]
        return x_index
    
    x_index = [*(indices[0] * [0]), *(indices[1] * [1])]
    return x_index

def index_arg_2D(indices: list = [[0,0], [0,0]],            # index AND coords of differentiation.
                 arg: int = 0):
    
    if not len(indices) == 2:
        raise('Wrong index dimensions for 2D. Please provide a list of length 2.')

    if arg == 0:
        x_index = [*((indices[0][0] + indices[0][1]) * [1]),                # number of times to differentiate wrt. `x`
                *((indices[1][0] + indices[1][1]) * [0])]                   # number of times to differentiate wrt. `z`
        x_coords = [*(indices[0][0] * [0]), *(indices[0][1] * [1])]         # first coordinate vs. second coordinate of `x`
        z_coords = [*(indices[1][0] * [0]), *(indices[1][1] * [1])]         # first coordinate vs. second coordinate of `z` 
        coords = [*x_coords, *z_coords]                                     # All coords stacked together. 
        return x_index, coords
    
    x_index = [*((indices[1][0] + indices[1][1]) * [1]),                # number of times to differentiate wrt. `z`
            *((indices[0][0] + indices[0][1]) * [0])]                   # number of times to differentiate wrt. `x`
    x_coords = [*(indices[0][0] * [0]), *(indices[0][1] * [1])]         # first coordinate vs. second coordinate of `x`
    z_coords = [*(indices[1][0] * [0]), *(indices[1][1] * [1])]         # first coordinate vs. second coordinate of `z` 
    coords = [*z_coords, *x_coords]                                     # All coords stacked together. 
    return x_index, coords

#==================================
# Isserlis generic
#==================================
def IsserlisPow2(covariance: Tensor, x2x2: bool = True): 
    """ Cov[Y^2(x), Y^2(x')] = 2 k (x, x')^2 """
    if x2x2:
        return 2 * covariance**2
    else: 
        return 0 * covariance                               # Cov[Y^2(x), Y(x')] = 0 for all x, x'

#==================================
# Isserlis non-stationary versions
#==================================
def IsserlisPow12(covariance: Tensor, 
                  mean: float | Tensor = 0.0):
    return 2 * mean * covariance

def IsserlisPow22(covariance: Tensor,
                  mean: float | Tensor = 0.0):
    return (2 * covariance**2) + (4 * mean**2 * covariance) 


#==================================
# RBF kernel as `nn.Module` classes
#==================================
DEF: Tensor = torch.tensor([1.0])                           # Default constant for lengthscale

class RBFKernel(nn.Module):
    """
    k(x,z) = exp(-||x-z||² / (2 lengthscale²)).
    
    Pass optional kwargs `diag: bool = True | False` and `DiffMode: bool = True | False`. 

    Both are `False` by default.
    """

    def __init__(self, 
                 lengthscale: Tensor = DEF,                 # Defaults to torch(1.0)
                 **kwargs) -> None:
        super().__init__()

        diag = False
        if 'diag' in kwargs:                                  
            diag = kwargs.get('diag')    

        DiffMode = False                                        
        if 'DiffMode' in kwargs:
            DiffMode = kwargs.get('DiffMode')
        
        self.diag_mode: bool = diag                         # diag mode or not.

        self.raw_lengthscale = nn.Parameter(
                    _inverse_softplus(lengthscale)          # Parameter to optimize using _inverse_softplus
        )                

        self.diff_mode: bool = DiffMode                     # works with expanded inputs to enable matrix derivatives
    
    @property
    def lengthscale(self) -> Tensor:
        return F.softplus(self.raw_lengthscale) + _EPS

    # Default lengthscale setter
    @lengthscale.setter
    def lengthscale(self, value):
        self.raw_lengthscale = nn.Parameter(
            _inverse_softplus(value)
        )

    def forward(self, x: Tensor, z: Tensor | None = None,
                ) -> Tensor:

        if (torch.equal(self.lengthscale, DEF) 
            and 
            not self.lengthscale.size(dim=-1) == x.size(dim=-1)
        ):      
                # Default lenthscale setter
                self.lengthscale = torch.ones(x.size(dim=-1))

        if not self.diff_mode:
            z = x if z is None else z
            d2 = weighted_squared_distance(x, z, 
                                           weights=self.lengthscale,
                                           diag=self.diag_mode)
            
            return torch.exp(-d2 / 2.0)

        if not self.diag_mode:
            z = x.transpose(0, 1) if z is None else z                   # Expanded inputs.
        else: 
            z = x if z is None else z                                   # Unexpanded inputs. diag_mode = True.     
        d2 = expanded_weighted_squared_distance(x, z, weights=self.lengthscale) 
        return torch.exp(-d2 / 2.0)

#==================================
# Additive RBF kernels as `nn.Module` classes
#==================================

class AdditiveRBFKernel(nn.Module):
    """
    k(x,z) = exp(-||x-z||² / (2 lengthscale1²)) + exp(-||x-z||² / (2 lengthscale2²)).
    
    Pass optional kwargs `diag: bool = True | False` and `DiffMode: bool = True | False`. 

    Both are `False` by default.
    """

    def __init__(self, 
                 lengthscale1: Tensor = DEF,                 # Defaults to torch(1.0)
                 lengthscale2: Tensor = DEF,                 # Defaults to torch(1.0)   
                 **kwargs) -> None:
        super().__init__()

        diag = False
        if 'diag' in kwargs:                                  
            diag = kwargs.get('diag')    

        DiffMode = False                                        
        if 'DiffMode' in kwargs:
            DiffMode = kwargs.get('DiffMode')
        
        self.diag_mode: bool = diag                         # diag mode or not.

        self.raw_lengthscale1 = nn.Parameter(
                    _inverse_softplus(lengthscale1)          # Parameter to optimize using _inverse_softplus
        )
        self.raw_lengthscale2 = nn.Parameter(
                    _inverse_softplus(lengthscale2)          # Parameter to optimize using _inverse_softplus
        )


        self.diff_mode: bool = DiffMode                     # works with expanded inputs to enable matrix derivatives
    
    @property
    def lengthscale1(self) -> Tensor:
        return F.softplus(self.raw_lengthscale1) + _EPS

    # Default lengthscale setter
    @lengthscale1.setter
    def lengthscale1(self, value):
        self.raw_lengthscale1 = nn.Parameter(
            _inverse_softplus(value)
        )

    @property
    def lengthscale2(self) -> Tensor:
        return F.softplus(self.raw_lengthscale2) + _EPS
    
    # Default lengthscale setter
    @lengthscale2.setter
    def lengthscale2(self, value):
        self.raw_lengthscale2 = nn.Parameter(
            _inverse_softplus(value)
        )

    def forward(self, x: Tensor, z: Tensor | None = None,
                ) -> Tensor:

        if (torch.equal(self.lengthscale1, DEF) 
            and 
            not self.lengthscale1.size(dim=-1) == x.size(dim=-1)
        ):      
                # Default lenthscale setter
                self.lengthscale1 = torch.ones(x.size(dim=-1))

        if (torch.equal(self.lengthscale2, DEF) 
            and 
            not self.lengthscale2.size(dim=-1) == x.size(dim=-1)
        ):      
                # Default lenthscale setter
                self.lengthscale2 = torch.ones(x.size(dim=-1))

        if not self.diff_mode:
            z = x if z is None else z
            d21 = weighted_squared_distance(x, z, 
                                           weights=self.lengthscale1,
                                           diag=self.diag_mode)
            d22 = weighted_squared_distance(x, z, 
                                            weights=self.lengthscale2,
                                            diag=self.diag_mode)
            
            return 0.5 * torch.exp(-d21 / 2.0) + 0.5 * torch.exp(-d22 / 2.0)
            # return torch.exp(-(d21 + d22) / 2.0)
        if not self.diag_mode:
            z = x.transpose(0, 1) if z is None else z               # Expanded inputs.
        else: 
            z = x if z is None else z                               # Unexpanded inputs. diag_mode = True.     
        d21 = expanded_weighted_squared_distance(x, z, weights=self.lengthscale1)
        d22 = expanded_weighted_squared_distance(x, z, weights=self.lengthscale2)
        return 0.5 * torch.exp(-d21 / 2.0) + 0.5 * torch.exp(-d22 / 2.0)
        # return torch.exp(-(d21 + d22) / 2.0)

#==================================
# Mat`ern 3/2 kernel as `nn.Module` classes
#==================================

class Matern32Kernel(nn.Module):
    """
    k(x,z) = (1 + sqrt{3} r) exp(-sqrt{3}r),

    where r = sqrt{||x-z||²/lengthscale²}
    
    Pass optional kwargs `diag: bool = True | False` and `DiffMode: bool = True | False`. 

    Both are `False` by default.
    """

    def __init__(self, 
                 lengthscale: Tensor = DEF,                 # Defaults to torch(1.0)
                 **kwargs) -> None:
        super().__init__()

        diag = False
        if 'diag' in kwargs:                                  
            diag = kwargs.get('diag')    

        DiffMode = False                                        
        if 'DiffMode' in kwargs:
            DiffMode = kwargs.get('DiffMode')
        
        self.diag_mode: bool = diag                         # diag mode or not.

        self.raw_lengthscale = nn.Parameter(
                    _inverse_softplus(lengthscale)          # Parameter to optimize using _inverse_softplus
        )                

        self.diff_mode: bool = DiffMode                     # works with expanded inputs to enable matrix derivatives
    
    @property
    def lengthscale(self) -> Tensor:
        return F.softplus(self.raw_lengthscale) + _EPS

    # Default lengthscale setter
    @lengthscale.setter
    def lengthscale(self, value):
        self.raw_lengthscale = nn.Parameter(
            _inverse_softplus(value)
        )

    def forward(self, x: Tensor, z: Tensor | None = None,
                ) -> Tensor:
        
        _SQRT3 = math.sqrt(3.0)

        if (torch.equal(self.lengthscale, DEF) 
            and 
            not self.lengthscale.size(dim=-1) == x.size(dim=-1)
        ):      
                # Default lenthscale setter
                self.lengthscale = torch.ones(x.size(dim=-1))

        if not self.diff_mode:
            z = x if z is None else z
            d2 = weighted_squared_distance(x, z, 
                                           weights=self.lengthscale,
                                           diag=self.diag_mode)
            d = torch.sqrt(d2 + _EPS)                           # _EPS to avoid division by zero when computing gradient.                  
            return (1 + (_SQRT3 * d)) * torch.exp(-(_SQRT3 * d))

        if not self.diag_mode:
            z = x.transpose(0, 1) if z is None else z               # Expanded inputs.
        else: 
            z = x if z is None else z                               # Unexpanded inputs. diag_mode = True. 

        d2 = expanded_weighted_squared_distance(x, z, weights=self.lengthscale) 
        d = torch.sqrt(d2 + _EPS)                               # _EPS to avoid division by zero when computing gradient.
        return (1 + (_SQRT3 * d)) * torch.exp(-(_SQRT3 * d))

#==================================
# Mat`ern 5/2 kernel as `nn.Module` classes
#==================================

class Matern52Kernel(nn.Module):
    """
    k(x,z) = (1 + sqrt{5} r + 5/3 r²) exp(-sqrt{5}r),

    where r = sqrt{||x-z||²/lengthscale²}
    
    Pass optional kwargs `diag: bool = True | False` and `DiffMode: bool = True | False`. 

    Both are `False` by default.
    """

    def __init__(self, 
                 lengthscale: Tensor = DEF,                 # Defaults to torch(1.0)
                 **kwargs) -> None:
        super().__init__()

        diag = False
        if 'diag' in kwargs:                                  
            diag = kwargs.get('diag')    

        DiffMode = False                                        
        if 'DiffMode' in kwargs:
            DiffMode = kwargs.get('DiffMode')
        
        self.diag_mode: bool = diag                         # diag mode or not.

        self.raw_lengthscale = nn.Parameter(
                    _inverse_softplus(lengthscale)          # Parameter to optimize using _inverse_softplus
        )                

        self.diff_mode: bool = DiffMode                     # works with expanded inputs to enable matrix derivatives
    
    @property
    def lengthscale(self) -> Tensor:
        return F.softplus(self.raw_lengthscale) + _EPS

    # Default lengthscale setter
    @lengthscale.setter
    def lengthscale(self, value):
        self.raw_lengthscale = nn.Parameter(
            _inverse_softplus(value)
        )

    def forward(self, x: Tensor, z: Tensor | None = None,
                ) -> Tensor:
        
        _SQRT5 = math.sqrt(5.0)

        if (torch.equal(self.lengthscale, DEF) 
            and 
            not self.lengthscale.size(dim=-1) == x.size(dim=-1)
        ):      
                # Default lenthscale setter
                self.lengthscale = torch.ones(x.size(dim=-1))

        if not self.diff_mode:
            z = x if z is None else z
            d2 = weighted_squared_distance(x, z, 
                                           weights=self.lengthscale,
                                           diag=self.diag_mode)
            d = (d2 + _EPS)**0.5                                        # _EPS to avoid division by zero when computing gradient.
            return (1 + _SQRT5 * d + (5.0 / 3.0) * d2) * torch.exp(-_SQRT5 * d)

        if not self.diag_mode:
            z = x.transpose(0, 1) if z is None else z               # Expanded inputs.
        else: 
            z = x if z is None else z                               # Unexpanded inputs. diag_mode = True. 
            
        d2 = expanded_weighted_squared_distance(x, z, weights=self.lengthscale) 
        d = (d2 + _EPS)**0.5                                            # _EPS to avoid division by zero when computing gradient.
        return (1 + _SQRT5 * d + (5.0 / 3.0) * d2) * torch.exp(-_SQRT5 * d)

#==================================
# Non-stationary kernel taken from https://arxiv.org/pdf/1703.10230 (Eq. 20)
#==================================

class NeuralNetworkCovariance(nn.Module):
    """
    I am using the form given in https://gaussianprocess.org/gpml/chapters/RW4.pdf (Eq. 4.29)

    k(x,x';\theta) = \frac{2}{\pi} \sin^{-1} 
    \left( 
                    \frac{2(\sigma_0^2+\sigma^2 x x')} 
    {\sqrt{(1+2(\sigma_0^2+\sigma^2 x^2)) (1+2(\sigma_0^2+\sigma^2 {x'}^2))}} 
    \right)

    Pass optional kwargs `diag: bool = True | False` and `DiffMode: bool = True | False`. 
    
    Both are `False` by default.
    """

    def __init__(self, 
                 sigma0: float = DEF,                 # Defaults to torch([1.0])
                 sigma: Tensor = DEF,                 # Defaults to torch([1.0])
                 **kwargs) -> None:
        super().__init__()


        diag = False
        if 'diag' in kwargs:                                  
            diag = kwargs.get('diag')    
        
        DiffMode = False                                        
        if 'DiffMode' in kwargs:
            DiffMode = kwargs.get('DiffMode')
        
        self.diag_mode: bool = diag                         # diag mode or not.
        
        self.raw_sigma0 = nn.Parameter(
                                    sigma0          # Parameter to optimize 
        )
        self.raw_sigma = nn.Parameter(
                                    sigma          # Parameter to optimize 
        )               
        
        self.diff_mode: bool = DiffMode                     # works with expanded inputs to enable matrix derivatives

    @property
    def sigma(self) -> Tensor:
        return self.raw_sigma**2 + _EPS
        
    # Default setter
    @sigma.setter
    def sigma(self, value):
        self.raw_sigma = nn.Parameter(
                                    value
        )
    
    @property
    def sigma0(self) -> Tensor:
        return self.raw_sigma0**2 + _EPS
    
    # Default setter
    @sigma0.setter
    def sigma0(self, value):
        self.raw_sigma0 = nn.Parameter(
                                    value
        )
        
    def forward(self, x: Tensor, z: Tensor | None = None,) -> Tensor:

        z = x if z is None else z
        if (torch.equal(self.sigma, DEF) 
            and 
            not self.sigma.size(dim=-1) == x.size(dim=-1)
        ):      
                # Default setter
                self.sigma = torch.ones(x.size(dim=-1))

        if not self.diff_mode:
            kernel = (((self.sigma * x) @ (self.sigma * z).T) + self.sigma0**2)
            num = 2 * kernel
            normalizer_1 = 1 + 2 * (((self.sigma * x) @ (self.sigma * x).T) + self.sigma0**2)
            normalizer_2 = 1 + 2 * (((self.sigma * z) @ (self.sigma * z).T) + self.sigma0**2)
            den = torch.sqrt(normalizer_1 * normalizer_2 + _EPS)
            return (2 / math.pi) * torch.arcsin(num / den)

        if not self.diag_mode:
            z = x.transpose(0, 1) if z is None else z               # Expanded inputs.
        else: 
            z = x if z is None else z                               # Unexpanded inputs. diag_mode = True. 

        num = 2 * ((self.sigma**2 * (x * z)).sum(dim=-1) + self.sigma0**2)
        normalizer_1 = 1 + 2 * ((self.sigma**2 * (x * x)).sum(dim=-1) + self.sigma0**2)
        normalizer_2 = 1 + 2 * ((self.sigma**2 * (z * z)).sum(dim=-1) + self.sigma0**2)
        den = torch.sqrt(normalizer_1 * normalizer_2 + _EPS)
        return (2 / math.pi) * torch.arcsin(num / den)

#=========================
# Product kernel for diagnostics
#=========================

class ProductKernel(RBFKernel):
    """
    k(x, x') = (1 + x^2) (1 + x'^2) [f(x) f(x')]

    Pass optional kwargs `diag: bool = True | False` and `DiffMode: bool = True | False`. 

    Both are `False` by default.
    """

    def __init__(self, 
                 lengthscale: Tensor = DEF,                 # Defaults to torch(1.0)
                 **kwargs) -> None:
        super().__init__(lengthscale=lengthscale, **kwargs)

    def forward(self, x: Tensor, z: Tensor | None = None,
                ) -> Tensor:

        if (torch.equal(self.lengthscale, DEF) 
            and 
            not self.lengthscale.size(dim=-1) == x.size(dim=-1)
        ):      
                # Default lenthscale setter
                self.lengthscale = torch.ones(x.size(dim=-1))

        if not self.diff_mode:
            z = x if z is None else z

            xl = (x / self.lengthscale)
            zl = (z / self.lengthscale)

            if self.diag_mode:
                fx = 1 + xl**2
                fz = 1 + zl**2
                return (fx * fz).prod(dim=-1)

            xl_expanded = xl.expand(zl.size(dim=0), 
                                    xl.size(dim=0), 
                                    xl.size(dim=-1))
            
            zl_expanded = zl.expand(xl.size(dim=0), 
                                    zl.size(dim=0), 
                                    zl.size(dim=-1)).transpose(0, 1)
            
            fx = 1 + (xl_expanded**2)
            fz = 1 + (zl_expanded**2)
            return (fx * fz).prod(dim=-1)

        if not self.diag_mode:
            z = x.transpose(0, 1) if z is None else z               # Expanded inputs.
        else: 
            z = x if z is None else z                               # Unexpanded inputs. diag_mode = True.     

        xl = (x / self.lengthscale)
        zl = (z / self.lengthscale)

        fx = 1 + xl**2
        fz = 1 + zl**2

        return (fx * fz).prod(dim=-1)

# Function for kernel choice
def choose_kernel(kernel: Literal['RBF', 'Matern32', 'Matern52'] = 'RBF'):
    """ choose the co-Kriging kernel """
    if kernel == 'RBF':
        return RBFKernel
    elif kernel == 'Matern32':
        return Matern32Kernel
    elif kernel == 'Matern52':
        return Matern52Kernel
    else:
        raise ValueError('Unsupported kernel choice')


#=========================
# Covariance derivatives
#=========================

def kernel_derivatives_1D(x: Tensor, z: Tensor | None = None,
                        kernel: RBFKernel = RBFKernel,
                        index: list = [0,0],
                        arg: int = 0,
                        diag: bool = False,
                ) -> Tensor:
    """ 
    Kernel derivatives 1D. `index = [x_order, z_order]`. Differentiate wrt. `x` first if `arg==0`.

    Expects `x` and `z` to be returned from `expanded_inputs()`.  
    `kernel.diff_mode == True` is expected. 
    """

    if not x.ndim == 3 and not z.ndim == 3:
        raise('Please provide expanded matrix inputs. See expanded_inputs().') 
     
    if kernel.diff_mode == False:
        raise('Cannot differentiate with kernel.diff_mode == False.')
    
    _EPSGRAD: float = 1e-10
    x_indices = index_arg_1D(index, arg)                                     # differentiation order wrt. `x`.

    x_clean = x.clone().detach().requires_grad_(True)                        # Refreshing the computational graph

    # Storing original diag_mode
    original_val = kernel.diag_mode
    
    if diag:
        z = x if z is None else z
    
        # Enabling diagonal mode
        kernel.diag_mode = True
    
        if not torch.equal(x, z):
            _EPSGRAD = 0.0
    
    elif not diag:
        z = x.transpose(0,1) if z is None else z
        if not torch.equal(x.transpose(0,1), z):
            _EPSGRAD = 0.0

    z_clean = z.clone().detach().requires_grad_(True) + _EPSGRAD             # Refreshing the computational graph + SMALL EPSILON.

    covariance = kernel(x_clean, z_clean)                                    # Compute the covariance matrix.
    grad_outputs = torch.ones_like(covariance)                               # grad shape

    _cov_diff = covariance                                                   # Initializing the differentiated matrix.

    order: int = len(x_indices)                                              # Total order of differentiation.
    for index in x_indices:
        # print(f'At start of the loop {order=}')
        # print(f'BOOL value {bool(order-1)=}')                                     
        INPUT = x_clean if index else z_clean
        _cov_diff = torch.autograd.grad(_cov_diff, INPUT,
                                        grad_outputs=grad_outputs,           # Grad shape
                                        create_graph=bool(order - 1),
                                        retain_graph=bool(order),)[0].squeeze(dim=-1)     # The slicing of element `[0]` only works for 1D.
        # print(f'At end of the loop {order=}')
        order = int(order - 1)                                                            # decrement the order variable

    # Resetting the original value diag value of the kernel
    kernel.diag_mode = original_val

    try:
        assert covariance.size() == _cov_diff.size()                            # basic sanity check
    except: 
        raise AssertionError(f'The differentiated covariance has size {_cov_diff.size()} vs. the original covariance had size {covariance.size()}'
                        ) 
    return _cov_diff

#=================================
# Derivatives in 2D
#=================================

def kernel_derivatives_2D(x: Tensor, z: Tensor | None = None,
                        kernel: RBFKernel = RBFKernel,
                        index:list =  [[0,0],
                                        [0,0]],
                        arg: int = 0,
                        diag: bool = False,
                        *outputs: int
                ) -> Tensor:
    """ 
    Kernel derivatives 2D. `index = [x_order, z_order]`. Differentiate wrt. `x` first if `arg==0`.
    
    Expects `x` and `z` to be returned from `expanded_inputs()`.  
    `kernel.diff_mode == True` is expected.

    :args: `*outputs` results from the computational graph. To avoid recalculation.
    """

    if not x.ndim == 3 and not diag:
        raise('Please provide expanded matrix inputs. See expanded_inputs().') 
     
    if kernel.diff_mode == False:
        raise('Cannot differentiate with kernel.diff_mode == False.')

    _EPSGRAD: float = 1e-10
    x_indices, coords = index_arg_2D(index, arg)                             # differentiation order wrt. `x` AND the coordinates among x_1, x_2, z_1, z_2. 
    x_clean = x.clone().detach().requires_grad_(True)                        # Refreshing the computational graph

    # Storing original diag_mode
    original_val = kernel.diag_mode

    if diag:
        z = x if z is None else z

        # Enabling diagonal mode
        kernel.diag_mode = True

        if not torch.equal(x, z):
            _EPSGRAD = 0.0

    elif not diag:
        z = x.transpose(0,1) if z is None else z
        if not torch.equal(x.transpose(0,1), z):
            _EPSGRAD = 0.0

    z_clean = z.clone().detach().requires_grad_(True) + _EPSGRAD             # Refreshing the computational graph + SMALL EPSILON.

    covariance = kernel(x_clean, z_clean)                                    # Compute the covariance matrix.
    grad_outputs = torch.ones_like(covariance)                               # grad shape
    cov_diff = covariance                                                    # Initializing the differentiated matrix.

    order: int = len(x_indices)                                              # Total order of differentiation.

    count: int = 0                                                           # Keeping count of the computational graph.
    OUTPUTS = []                                                             # OUTPUTS to save some results.
    for  index, coord in zip(x_indices, coords):                             # coords among 0 or 1.

        INPUT = x_clean if index else z_clean

        if count in outputs:
            OUTPUTS.append(cov_diff)                                         # Add the result if asked for in *outputs

        if not diag:
            cov_diff = torch.autograd.grad(cov_diff, INPUT,
                                grad_outputs=grad_outputs,
                                create_graph=bool(order - 1),
                                retain_graph=bool(order))[0][:,:,coord]      # Appropriate slicing
        elif diag:
            cov_diff = torch.autograd.grad(cov_diff, INPUT,
                                grad_outputs=grad_outputs,
                                create_graph=bool(order - 1),
                                retain_graph=bool(order))[0][:,coord]        # Appropriate slicing
        
        order = int(order - 1)                                               # decrement the order variable
        count = int(count + 1)                                               # Going forward in the computational graph.

    # Resetting the original value diag value of the kernel
    kernel.diag_mode = original_val

    if OUTPUTS:                                                              # If list is non-empty.
        return *OUTPUTS, cov_diff                                            # Output the relevant derivatives. 
    return cov_diff    


def kernel_diff_1D_asymm(x: Tensor, z: Tensor | None = None,
                    kernel: RBFKernel = RBFKernel,
                    index: int = 0,
                    order: int = 0):

    _EPSGRAD: float = 1e-10
    # Ensure inputs track gradients
    # Cloning and detaching ensures we don't accidentally carry over old graph states from x
    x_clean = x.clone().detach().requires_grad_(bool((1-index)*True))        # The detach() ensures that `x` is detached from any previous computational graphs and starts afresh.  

    z = x if z is None else z

    if not torch.equal(x, z):
        _EPSGRAD = 0.0
    z_clean = z.clone().detach().requires_grad_(bool(index*True)) + _EPSGRAD  

    covariance = kernel(x_clean, z_clean)

    # ORDER = 0
    if order == 0:
        return covariance

    # ORDER = 1
    cov_xz_list = []
    if order == 1:
        # --- First Derivatives (create_graph=True is mandatory for 2nd derivatives) ---
        output_shape = torch.ones_like(covariance[:,0])                  # The output shape would be of this shape. 

        for _ in range(covariance.shape[1]):
            # print(covariance[:,_])
            # print((1-index) * x_clean + index * z_clean)
            INPUT = x_clean if not index else z_clean 
            # print(torch.autograd.grad(covariance[:,_],            # Column-wise differentiation
            #                             # (1-index) * x_clean + index * z_clean,
            #                             INPUT, 
            #                             grad_outputs=output_shape, 
            #                             create_graph=False, 
            #                             retain_graph=False,))[0]
            
            cov_xz_col = torch.autograd.grad(covariance[:,_],            # Column-wise differentiation
                                            # (1-index) * x_clean + index * z_clean,
                                            INPUT, 
                                            grad_outputs=output_shape, 
                                            create_graph=False, 
                                            retain_graph=True)[0][:,0]      
            cov_xz_list.append(cov_xz_col)

        # Stack list of tensors back into a matrix cleanly (avoiding in-place mutations)
        cov_xz = torch.stack(cov_xz_list, dim=1)

        return covariance, cov_xz

    # ORDER >= 2
    if order > 1:
        cov_diff = covariance                               # Initiate cov_diff

        while not order == 0:
            cov_diff_list = []
            output_shape = torch.ones_like(covariance[:,0])                  # The output shapes.

            for _ in range(covariance.shape[1]):
                cov_diff_col = torch.autograd.grad(cov_diff[:,_],            # Column-wise differentiation
                                                   (1-index) * x_clean + index * z_clean,
                                                    grad_outputs=output_shape,
                                                    create_graph=bool(order-1),
                                                    retain_graph=bool(order),
                                        )[0][:,0]
                cov_diff_list.append(cov_diff_col)

            # Stack list of tensors back into a matrix cleanly (avoiding in-place mutations)
            cov_diff = torch.stack(cov_diff_list, dim=1)

            # Reduce the order after each iteration
            order -= 1
        return cov_diff 

# ---------------------------------------------------------------------
# Kernel regression
# ---------------------------------------------------------------------

class KernelRegressor(nn.Module):
    """
    GP regression.
    """

    def __init__(
        self,
        kernel: RBFKernel,
        jitter: float = 1e-6,
    ) -> None:
        super().__init__()

        if jitter <= 0:
            raise ValueError("jitter must be positive")

        self.kernel = kernel
        self.jitter = float(jitter)

    def forward(
        self,
        x: Tensor,
        z: Tensor | None = None,
    ) -> Tensor:
        return self.kernel(x, z)

    def _training_cholesky(self, x: Tensor) -> Tensor:
        covariance = self.kernel(x)
        identity = torch.eye(
            x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )

        return torch.linalg.cholesky_ex(
            covariance
            + (self.jitter) * identity
        )

    def LOOCVloss(self, x: Tensor, y: Tensor, 
                to_center: bool = True) -> Tensor:
    
        y = y.reshape(-1)
        if x.ndim != 2 or len(y) != len(x):
            raise ValueError("Loss computation error - x and y must contain the same number of samples")
    
        cholesky = self._training_cholesky(x)
        inv = torch.cholesky_inverse(cholesky)
    
        diag2 = (torch.diag(torch.diagonal(inv)**(-2)))
    
        # Centering the observations
        if to_center:
            y_mean = torch.mean(y)            
            y_c = y - y_mean
        else:
            y_c = y
        
        y_c = y_c.reshape(-1).double()
    
        Ky = torch.linalg.solve_triangular(
                                cholesky,
                                y_c,
                                upper=False 
        )

        return (1 / x.size(dim=0)) * (Ky.T @ diag2 @ Ky)

    def NLLloss(self, x: Tensor, y: Tensor) -> Tensor:
        y = y.reshape(-1)
        if len(y) != len(x):                        # Removed: `x.ndim != 2 or`
            raise ValueError("Loss computation error - x and y must contain the same number of samples")
        cholesky, info = self._training_cholesky(x)

        alpha = torch.cholesky_solve(
            y[:, None], cholesky
        )[:, 0]

        data_fit = 0.5 * y.dot(alpha)
        complexity = torch.log(
            torch.diagonal(cholesky)
        ).sum()
        constant = 0.5 * len(y) * math.log(2.0 * math.pi)

        return (data_fit + complexity + constant) / len(y)

    def negative_log_marginal_likelihood(
        self, x: Tensor, y: Tensor
    ) -> Tensor:
        """Descriptive alias used by the training example."""
        return self.NLLloss(x, y)

    @torch.no_grad()
    def predict(
        self,
        x_train: Tensor,
        y_train: Tensor,
        x_test: Tensor,
        x_train_test: Tensor | None = None,
        x_test_diagonal: Tensor | None = None
        # include_noise: bool = False,
    ) -> tuple[Tensor, Tensor]:
        y_train = y_train.reshape(-1)
        cholesky, info = self._training_cholesky(x_train)

        alpha = torch.cholesky_solve(
            y_train[:, None], cholesky
        )[:, 0]

        if self.kernel.diff_mode == False:
            cross_covariance = self.kernel(x_test, x_train)
        else:
            if x_train_test == None:
                raise('Please provide `x_train_test` in DiffMode=True.')
            cross_covariance = self.kernel(x_test, x_train_test)
                
        prediction = cross_covariance @ alpha

        solved = torch.linalg.solve_triangular(             # Solve LX = H, LL.T = K :  X = L^{-1} H, X.T X    
            cholesky,                                       # = H.T L^{-1}.T L^{-1} H 
            cross_covariance.T,                             # = H.T K^{-1} H
            upper=False,
        )
        if self.kernel.diff_mode == False:
            prior_variance = torch.diagonal(self.kernel(x_test))
        else: 
            if x_test_diagonal == None:
                raise('Please provide `x_test_diagonal` in DiffMode=True.')
            prior_variance = torch.diagonal(self.kernel(x_test_diagonal))

        variance = (
            prior_variance - solved.square().sum(dim=0)
        ).clamp_min(0.0)

        return prediction, variance.sqrt()

#=======================
# Otpimization routine
#=======================

def fit_regressor(
    model: KernelRegressor,
    x_train: Tensor,
    y_train: Tensor,
    steps: int = 500,
    learning_rate: float = 0.01,
    interactive: bool = False,
) -> list[float]:
    """Optimize only the parameters contained in the kernel."""

    losses: list[float] = []

    if steps < 1:
        raise ValueError("steps must be positive")

    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")

    if list(model.parameters()) == []:
        print('Model has no parameters to optimize.')                   # LinearKernel(), for example.
        losses = steps*[model.NLLloss(x_train, y_train)]
        return losses
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate) 

    model.train()
    for _, _num  in zip(
        tqdm(range(steps), colour = 'green', position=0),
        range(steps)
    ):
        optimizer.zero_grad(set_to_none=True)
        loss = model.NLLloss(x_train, y_train)

        if interactive:
            print(f'Negative log-liklihood loss at step {_num}/{steps} - {loss.item():.2f}', 
            end='\r', flush=True)

        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss encountered during training")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        losses.append(loss.detach().item())

    model.eval()
    return losses

#=========================
# Physics-based regressor
#=========================
def general_fit_regressor(
        model: KernelRegressor,
        x_train: Tensor,
        z_train:  Tensor,
        y_train: Tensor,
        v_train: Tensor | None = None, 
        lossfn: str = 'LOOCV',                                      
        steps: int = 500,
        learning_rate: float = 0.01,
        interactive: bool = False,
        **kwargs 
) -> list[float]:

    """Optimize only the parameters contained in the kernel."""

    lengthscales: list[Tensor] = []
    gradients: list[Tensor] = []
    losses: list[float] = []
    
    # if steps < 1:
    #     raise ValueError("steps must be positive")
    
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")

    if list(model.parameters()) == []:
        print('Model has no parameters to optimize.')                   # LinearKernel(), for example.
        losses = steps*[model.LOOCVloss(x_train, z_train, y_train, v_train, **kwargs)]
        return losses

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, 
                                 weight_decay=1e-4,
                        )

    model.train()

    # Saving the best hyperparameters
    best = None

    for _, _num  in zip(
        tqdm(range(steps), colour = 'green', position=0),
        range(steps)
    ):
        optimizer.zero_grad(set_to_none=True)
        loss = model.LOOCVloss(x_train, z_train, y_train, v_train, **kwargs)

        lengthscales.append(model.kernel.lengthscale.tolist())

        # compulsory pass in the loop
        if _num == 0:
            best = loss.item()                        # Initiate the best value
            bestparam = model.kernel.lengthscale      # Initiate the lengthscale

        # optional pass: if loss is lower, best = loss.
        if best >= loss.item():
            bestparam = model.kernel.lengthscale      # Saving the best lengthscale
            best = loss.item()                        # New best score

        if interactive:
            print(f'{model.kernel.lengthscale.tolist()} loss at step {_num}/{steps} - {loss.item():.2f}',) 
            # end='\r', flush=True)

        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss encountered during training")

        loss.backward()

        for param in model.parameters():
        #     gradients.append(param.grad)
            p_data = param.data
            p_grad = param.grad
        
            print(f'Automatic differentiation gradient approximate: {param.grad}')
        
            gradlist = []
            for idx in range(len(p_data)):
            #   idx = iterator.multi_index
              original_val = p_data[idx].item()
            
              # 1. Evaluate loss at x + eps
              p_data[idx] = original_val + 1e-6
              loss_high = model.LOOCVloss(x_train, z_train, y_train, v_train, **kwargs).item()
            
              # 2. Evaluate loss at x - eps
              p_data[idx] = original_val - 1e-6
              loss_low = model.LOOCVloss(x_train, z_train, y_train, v_train, **kwargs).item()
        
              gradlist.append(((loss_high - loss_low) / (2 * 1e-6)))
        
              p_data[idx] = original_val
            print(f'Central differences gradient approximate: {gradlist}')

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        losses.append(loss.detach().item())

    if  steps:
        print(f'Final lengthscale: {model.kernel.lengthscale.tolist()} and LOOCV score {loss.detach().item()}')
    if not best == None:  
        print(f'Best lengthscale {bestparam.tolist()} and best LOOCV score: {best}')
        model.kernel.lengthscale = bestparam 

    model.eval()
    return losses

def FD_fit_regressor(
        model: KernelRegressor,
        x_train: Tensor,
        z_train:  Tensor,
        y_train: Tensor,
        v_train: Tensor | None = None, 
        lossfn: str = 'LOOCV',                                      
        steps: int = 500,
        learning_rate: float = 0.01,
        interactive: bool = False,
        **kwargs 
) -> list[float]:

    """A finite differences implementation of the general_fit_regressor()."""

    lengthscales: list[Tensor] = []
    gradients: list[Tensor] = []
    losses: list[float] = []
    
    # if steps < 1:
    #     raise ValueError("steps must be positive")
    
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")

    if list(model.parameters()) == []:
        print('Model has no parameters to optimize.')                   # LinearKernel(), for example.
        losses = steps*[model.LOOCVloss(x_train, z_train, y_train, v_train, **kwargs)]
        return losses

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, 
                                 weight_decay=1e-4,
                        )

    model.train()

    # Saving the best hyperparameters
    best = None

    for _, _num  in zip(
        tqdm(range(steps), colour = 'red', position=0),
        range(steps)
    ):
        optimizer.zero_grad(set_to_none=True)
        loss = model.LOOCVloss(x_train, z_train, y_train, v_train, **kwargs)

        # lengthscales.append(model.kernel.lengthscale.tolist())

        # compulsory pass in the loop
        if _num == 0:
            best = loss.item()                        # Initiate the best value
            bestparam = model.kernel.lengthscale      # Initiate the lengthscale

        # optional pass: if loss is lower, best = loss.
        if best >= loss.item():
            bestparam = model.kernel.lengthscale      # Saving the best lengthscale
            best = loss.item()                        # New best score

        if interactive:
            print(f'{model.kernel.lengthscale.tolist()} loss at step {_num}/{steps} - {loss.item():.2f}',) 
            # end='\r', flush=True)

        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss encountered during training")

        for param in model.parameters():

        # Initialize gradient tensor if it doesn't exist
            if param.grad is None:
                param.grad = torch.zeros_like(param)

            p_data = param.data
            p_grad = param.grad
        
            gradlist = []
            for idx in range(len(p_data)):
            #   idx = iterator.multi_index
              original_val = p_data[idx].item()
            
              # 1. Evaluate loss at x + eps
              p_data[idx] = original_val + 1e-4
              loss_high = model.LOOCVloss(x_train, z_train, y_train, v_train, **kwargs).item()
            
              # 2. Evaluate loss at x - eps
              p_data[idx] = original_val - 1e-4
              loss_low = model.LOOCVloss(x_train, z_train, y_train, v_train, **kwargs).item()

              # modify p_grad
              p_grad[idx] = ((loss_high - loss_low) / (2 * 1e-4))

            #   gradlist.append(((loss_high - loss_low) / (2 * 1e-4)))
        
              p_data[idx] = original_val
            # print(f'Central differences gradient approximate: {gradlist}')

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        losses.append(loss.detach().item())

    if  steps:
        print(f'Final lengthscale: {model.kernel.lengthscale.tolist()} and LOOCV score {loss.detach().item()}')
    if not best == None:  
        print(f'Best lengthscale {bestparam.tolist()} and best LOOCV score: {best}')
        model.kernel.lengthscale = bestparam 

    model.eval()
    return losses

#******************************
# Main body (OPTIONAL)
#******************************

################
# Arguments
################

# Source - https://stackoverflow.com/a/715468
# Posted by Brian R. Bondy, modified by community. See post 'Timeline' for change history
# Retrieved 2026-08-21, License - CC BY-SA 4.0


# #===============================
# # My Krig

# N_rand = args.nobs

# u_true, x_train, c_train, x_test, y_train, v_train = Burgers_DoE(N_colloc=args.nc,
#                                                                   nu = args.nu, N_rand=N_rand, 
#                                                                   seed = args.seed, t_obs=args.tobs)

# Krig = KernelRegressor(kernel = RBFKernel(lengthscale = torch.tensor(thetaSK)),
#                         jitter = 1e-10)                                         # Basic Kriging

# print(f'LOOCV loss loss :  {Krig.LOOCVloss(x_train, 
#                                             y_train)}')

# likelihood = gpt.likelihoods.GaussianLikelihood()
# model = ExactGPModel(x_train, y_train, likelihood, 
#                      output_scale=torch.tensor([1.0]),
#                      lengthscale = torch.tensor(thetaSK)
#             )
# output = model(x_train)

# loocv = MyLeaveOneOutPseudoLikelihood(likelihood, model)

# print(f'For noise var = 1.0, LOOCV loss: {-loocv(output, y_train)}')


# #===============================

#************************
# Main body
#************************

def main():

    generator = torch.Generator().manual_seed(42)
    from OLDnonlinearcodefiles.covderivatives import torchdiff1D
    import numpy as np

    x1 = torch.rand(5, 1, generator=generator)

    x1_symm = expanding_inputs(x1, grad = True)

    _kernel = RBFKernel(lengthscale=torch.tensor([2.0]),                            # Arbitrary lengthscale
                        DiffMode = True,                                            # `DiffMode` activated         
        )                                            

    cov_diff = kernel_derivatives_1D(x1_symm, None,
                                    _kernel, index = [1,1])

    print(f'pytorch implementation prediction = \n {cov_diff}')
    print(50*'%'+'\n Numpy 1D check using torchdiff1D(): \n'+50*'%')

    x1_np = x1.detach().numpy()

    K = np.zeros((len(x1_np), len(x1_np)))                                            # initialize
    for _j in range(len(x1_np)):
        for _i in range(len(x1_np)):
            K[_j, _i] =  torchdiff1D(x1_np[_i], x1_np[_j], 
                                               xindex = 1, 
                                               yindex = 1,
                                            thetaval = 2.0)

    print(K)
    

if __name__ == "__main__":
    main()



