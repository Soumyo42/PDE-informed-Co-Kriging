# Physics informed co-Kriging (general capabilities -- change the methods for each problem)

# imports
import os
import torch

from kernels import RBFKernel, expanding_inputs, printRed, symmetry_check, dim_check
from SimpleKrigGpytorch import fit_regressor as gpytorch_regressor
from SimpleKrigGpytorch import MyLeaveOneOutPseudoLikelihood
import gpytorch

from torch import Tensor, nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
from typing import Callable

# Useful functions

# Co-Kriging flash
#===============
# Colored text
#===============
BLUE, END = '\033[94m', '\033[0m'
printBlue = lambda sTxt: print(BLUE + sTxt + END)

def _CK_flash():
    printBlue(100*'=')
    print('Co-Kriging implementation from pytorch')
    printBlue(100*'=')

# Constructing the adaptive nugget with the trace normalizer
def _adaptive_nugget_constructor(*args):
    """ provide the block diagonal entries of a symmetric block matrix """
    _normalizer = torch.trace(args[0])
    blocks = []
    for arg in args:
        factor = torch.trace(arg) / _normalizer
        blocks.append(
            factor * torch.eye(
                arg.shape[0], dtype = arg.dtype
                )
        )
    return torch.block_diag(*blocks)

# Symmetric block matrix constructor from lower triangular
def _symmetric_block(*blocks):
    """ construct a SYMMETRIC block matrix from just the lower triangular entries (saves time) """
    r = len(blocks)

    # Simple quadratic formula to find total number of rows n that solves: (n^2 - n)/2 + n = r
    n = int((
        ((1 + 8 * r)**(0.5)) - 1
        ) / 2)
    # N = int(n**2)                       # square matrix               
    
    rows = []

    # row constuctor
    # print(f'number of total elements = {N}')
    for integer in range(1, n+1):
        # print(f'row number {integer}')
        row = []
        for _i in range(int(integer * (integer - 1) / 2), 
                        int(integer * (integer + 1) / 2)
                    ):
            row.append(blocks[_i])
            # print(f'joining blocks {_i}')
        for _j in range(integer, 
                        n):
            row.append(blocks[int(
                (
                    (_j * (_j + 1)) / 2) 
                    + integer - 1
                    )].T)
            # print(f'transpose joining blocks {int(
            #             ((_j * (_j + 1)) / 2) 
            #             + integer - 1
            #             )}'
            #     )

        rows.append(torch.cat(row, dim = 1))

    return torch.cat(rows)

# Creating a physics-matrix class
class PhysicsMatrix():
    """ Matrix class with methods like cholesky decomposition - trace computation - etc. """
    def __init__(
        self,
        matrix: Tensor,                                                 # Usually the co-Kriging observation matrix
        nugget: Tensor | None = None,                                   # Adaptive nugget or identity
        diag_blocks: list | None = None,                                # diagonal blocks     
        interactive: bool = False,                                      # to print intermediary values
        time_enabled: bool = False                                      # Whether to time specific methods or not
    ) -> None:
        super().__init__()

        self.matrix = matrix.double()
        self.nugget = nugget.double() 
        self.interactive = bool(interactive)
        self.diag_blocks = diag_blocks if diag_blocks is not None else None

    def forward(self):
        return self.matrix

    def printInteractive(self, string: str):
        if self.interactive:
            print(string) 

    # Matrix inversion in pytorch.
    def invert(self, nugget: float = 1e-10, mode: str = 'pinv'):
        self.printInteractive('Inverting ...')
        if mode == 'solve':
            self.printInteractive('Using `torch.linalg.solve` for invert ...')
            return torch.linalg.solve(self.matrix + (nugget * self.nugget), 
                                  torch.eye(self.matrix.shape[0], dtype = self.matrix.dtype))
        elif mode == 'inv':
            self.printInteractive('Using `torch.linalg.solve` for invert ...')
            return torch.linalg.inv(self.matrix + (nugget * self.nugget))

        self.printInteractive('Using `torch.linalg.pinv` for invert ...')
        return torch.linalg.pinv(self.matrix + (nugget * self.nugget))

    # Cholesky decomposition in pytorch.
    def cholesky(self, jitter: float = 1e-6, cholesky_order: bool = False):
        self.printInteractive('Computing cholesky decomposition ...')
        cholesky, info = torch.linalg.cholesky_ex(self.matrix + (jitter * self.nugget))

        if cholesky_order and not self.interactive:
            print('Cholesky order')
        self.printInteractive('Cholesky order')

        return cholesky 

    # Eigenvalues of the matrix [TODO]
    def eigen(self, jitter: float = 1e-6):
        return torch.linalg.eig(self.matrix + (jitter * self.nugget))

    # The size of each diagonal block as a tensor
    def size_diags(self):
        sizes = []
        for diag in self.diag_blocks:
            sizes.append(diag.size(dim=0))
        return torch.tensor(sizes)
    
# Centered observations
class CenteredValues():
    """ Centering functions and Tensor that stores all of the centered information (the centering, the centered output, etc.) """
    def __init__(
        self,
        kernel: RBFKernel = RBFKernel,                                  # kernel function to compute with.                                                                                      
        to_center: bool = True,                                         # To center observations or not
        time_enabled: bool = False                                      # Whether to time specific methods or not
    ) -> None:
        super().__init__()

        self.to_center = bool(to_center)
        self.kernel = kernel

    # Simple forward function to save all the information
    def forward(self, y: Tensor, _centering: Tensor | None = None,):
        self.uncentered = y.double()
        self.centering = _centering.double() if torch.is_tensor(_centering) else _centering
        if self.to_center and _centering is not None:
            self.centered = (y - _centering).double()
        else:
            self.centered = y.double()
            self.centering = 0.0

    #======================
    # Problem specific
    #======================
    # Centering function
    def centering_x(self, y: Tensor, x: Tensor | None = None):
        """ Centering the given observations (OF Y OR Y^2). """

        if x is None:
            return CenteredValues.forward(y, torch.mean(y))
    
        elif x is not None:
            # Assumes input is squared process
            original_val = self.kernel.diag_mode 
            self.kernel.diag_mode = True
    
            # Slightly faster
            y2_centering = self.kernel(x)
    
            self.kernel.diag_mode = original_val
            return self.forward(y, y2_centering)
    
    #======================
    # Problem specific
    #======================
    # Centering function at collocation points (based on physics)
    def centering_z(self, v: Tensor, z: Tensor | None = None):
        if z is None:
            return self.forward(v)

    # Full centered observations
    def ConcatenatedObs(self, y: Tensor, 
                 x: Tensor | None = None, 
                 v: Tensor | None = None,
                 z: Tensor | None = None):

        Y_TRAIN = self.centering_x(y, x)
        V_TRAIN = self.centering_z(v, z)

        ConcatedObs = torch.cat([Y_TRAIN.centered, 
                                 V_TRAIN.centered]).reshape(-1).double()

        return ConcatedObs, Y_TRAIN.centering


# Creating a very generic physics class
class PhysicsRegression(nn.Module):
    r"""
    Generic physics based co-Kriging class. 

    Define the following methods:
        1. `.forward()`: this is going to give the observation covariance matrix, \Sigma.
        2. `._cross_covariance(mode:str='u')`: define one or multiple modes, \Sigma_\ast. 
    """

    def __init__(
        self,
        kernel: RBFKernel,                                              # choose a kernel
        nu: float = 0.01/math.pi,                                       # visocity coefficient
        rho: float = 5,                                                 # reaction coefficient
        beta: int = 30,                                                 # convection coefficient
        jitter: float = 1e-6,                                           # torch.linalg.cholesky()
        nugget: float = 1e-10,                                          # torch.linalg.solve()
        mode: str = 'u2',                                               # Whether to use u^2 observations or not
        true_fn: Tensor | None = None,                                  # Pass the true function to compute RMSE
        time_enabled: bool = False,                                     # Whether to time specific methods or not
        save: bool = False,                                             # To save model to compute results later
    ):
        super().__init__()

        if jitter <= 0 or nugget <= 0:
            raise ValueError("jitter / nugget must be positive")

        self.kernel = kernel
        self.jitter = float(jitter)
        self.nugget = float(nugget)
        self.nu = float(nu)
        self.rho = float(rho)
        self.beta = float(beta)
        self.time_enabled = time_enabled  # The boolean flag
        self.true_fn = true_fn.double() if true_fn is not None else None
        self.mode: str = str(mode)         # normal mode or u2 mode           
        self.save: bool = bool(save)    

    #======================
    # Problem specific
    #======================
    def forward(
            self,
            x: Tensor,                                              # observations locations
            z: Tensor | None = None,                                # collocation locations
            adaptive_nugget: bool = False,                          # To compute an adaptive. nugget for stability: taken from Appendix A.1, https://arxiv.org/pdf/2103.12959.
            diag_blocks: bool = True                                # Pass the diag blocks to PhysicsMatrix
    ) -> PhysicsMatrix:
        """ returns the K matrix """
    
        xsymm = expanding_inputs(x)
        if z == None:
            printRed('provided with zero collocation points (simple Kriging K matrix)')
            return self.kernel(xsymm)
    
        zsymm = expanding_inputs(z)
        x_expanded, z_expanded = expanding_inputs(x, z)
    
        #--- First row ---
        K11 = self.kernel(xsymm)
    
        #--- Second row ---
        K21 = self.kernel(z_expanded, x_expanded)
        K22 = self.kernel(zsymm)

        CoK = _symmetric_block(K11, K21, K22)

        # Invoking adaptive nugget
        self._nugget = torch.eye(CoK.shape[0], dtype = CoK.dtype) 
        if adaptive_nugget:
            self._nugget = _adaptive_nugget_constructor(K11, K22)

        # Sanity check
        symmetry_check(CoK)

        # Sanity check
        dim_check(CoK, self._nugget)

        # Defining a physicsmatrix object
        K = PhysicsMatrix(CoK, self._nugget)
        if diag_blocks:
            K = PhysicsMatrix(CoK, 
                              self._nugget, 
                              [K11, K22])
        self.Kmatrix = K
        return K 

    #======================
    # Problem specific
    #======================
    # cross covariance matrix
    def cross_covariance(self,
                    x: Tensor,
                    x_test: Tensor,
                    z: Tensor | None = None,              
            )-> Tensor:
        """ returns the H matrix """

        x_t, t_x = expanding_inputs(x, x_test)
        H1 = self.kernel(x_t, t_x)

        if z == None:
            printRed('provided with zero collocation points (simple Kriging H matrix)')
            return H1
        
        z_t, t_z = expanding_inputs(z, x_test)
        H2 = self.kernel(z_t, t_z)
        CoH = torch.cat([H1, H2,], dim = 1).double()

        self.Hmatrix = CoH
        return CoH

    # Computing the error metrics
    def metrics(self, prediction, reference, 
                rel_tol: float | None = None) -> None:
        rmse = torch.mean((prediction.reshape(*reference.size()) - reference)**2)**0.5
        mae = torch.mean((torch.abs(prediction.reshape(*reference.size()) - reference)))

        # rel_tol = torch.mean(reference) if rel_tol is None else rel_tol
        rel_tol = 1e-2 if rel_tol is None else rel_tol          # avoid division by zero.
        rel_tol = 1e-2 if rel_tol <= 1e-2 else rel_tol

        l2relative = torch.linalg.norm((prediction.reshape(*reference.size()) - reference).flatten(), 
                                       ord = 2, dim = 0) / torch.linalg.norm(reference.flatten(), 
                                                                             ord = 2, dim = 0)  

        # l2relative = torch.mean((
        #     torch.abs(prediction.reshape(*reference.size()) - reference) / (
        #         torch.abs(reference) + rel_tol
        #         )
        #     )
        # )
        
        # Error attributes
        self.RMSE = rmse
        self.MAE = mae
        self.L2RELATIVE = l2relative
    
    # Add a kernel regression function (via cholesky decomposition)
    def predict_cholesky(self, cholesky, cross_covariance, 
                         centering: Tensor,
                         centered: Tensor,
                         test_points: Tensor | None = None,
                         base_var: bool = True,            # Making the conditional variance positive
                         min_val: float = 0.0,             # to clamp the conditional variance
                         sigma: float = 1.0,               # noise variance
                         ):
        # Sanity check
        dim_check(centered, cholesky, index = 0)

        alpha = torch.cholesky_solve(                      # Solve (LL^T) X = Y_TRAIN => X = K^{-1} Y_TRAIN  
            centered[:, None], cholesky
        )[:, 0]

        # Final predictions
        self.prediction = (cross_covariance @ alpha) + centering

        if self.true_fn is not None:
            # computing the error metrics  
            self.metrics(self.prediction, self.true_fn)

        solved = torch.linalg.solve_triangular(             # Solve LX = H, LL.T = K :  X = L^{-1} H, X.T X    
            cholesky,                                       # = H.T L^{-1}.T L^{-1} H 
            cross_covariance.T,                             # = H.T K^{-1} H
            upper=False,
        )

        with torch.no_grad():
            # Slightly faster
            original_val = self.kernel.diag_mode                                      
            self.kernel.diag_mode = True

            if test_points is not None:
                prior_variance = self.kernel(test_points)

                base_var = True                 # Always using base_var
                if base_var:
                    #=================
                    # old idea: add min possible base variance everywhere to make it positive
                    #=================
                    # min_val = torch.min(prior_variance - solved.square().sum(dim=0))
                    # neg_check = (1 - torch.sign(min_val))
                    # prior_variance += (torch.abs(min_val) + 1e-8) * neg_check 
 
                    #=================
                    # new idea via Didier: .clamp_min(0.1) for squared process
                    #=================
                    variance = sigma**2 * (
                        prior_variance - solved.square().sum(dim=0) 
                    ).clamp_min(min_val)           

            # Assertion check to see if solved.square().sum(dim=0) is the diag(H.T @ Kinv @ H)
            try: 
                assert torch.allclose(solved.square().sum(dim=0), 
                                  torch.diagonal(cross_covariance @ torch.cholesky_inverse(cholesky) @ cross_covariance.T),)
            except AssertionError:
                print(f'Max error in HTH computation with cholesky vs. inverse = {
                                            torch.max(
                                                torch.abs(solved.square().sum(dim=0) - 
                                                torch.diagonal(cross_covariance @ torch.cholesky_inverse(cholesky) @ cross_covariance.T))
                                        )}'
                    ) 

            self.kernel.diag_mode = original_val

            self.std_dev = variance.sqrt() if test_points is not None else None 

        return self.prediction, self.std_dev

    # Add a kernel regression function (via matrix inversion)
    def predict_inv(self, inverse, cross_covariance, 
                    centering: Tensor,
                    centered: Tensor,
                    test_points: Tensor | None = None,
                    base_var: bool = True,            # Making the conditional variance positive
                    min_val: float = 0.0,             # to clamp the conditional variance
                    sigma: float = 1.0,               # noise variance
                ):

        # Sanity check
        dim_check(centered, inverse, index = 0)

        # Final predictions
        intermediary = cross_covariance @ inverse
        self.prediction = (intermediary @ centered) + centering

        if self.true_fn is not None:
            # computing the error metrics  
            self.metrics(self.prediction, self.true_fn)

        HTH = intermediary @ cross_covariance.T

        with torch.no_grad():
            # Slightly faster
            original_val = self.kernel.diag_mode                                      
            self.kernel.diag_mode = True
        
            if test_points is not None:
                prior_variance = self.kernel(test_points).squeeze()

                if base_var:
                    min_val = torch.min(prior_variance - torch.diagonal(HTH))
                    neg_check = (1 - torch.sign(min_val))
                    prior_variance += (torch.abs(min_val) + 1e-4) * neg_check  

                variance = sigma**2 * (
                    prior_variance - torch.diagonal(HTH) 
                ).clamp_min(min_val)
        
            self.kernel.diag_mode = original_val
        
            self.std_dev = variance.sqrt() if test_points is not None else None 
        
        return self.prediction, self.std_dev

    # search for best jitter
    def search_best_jitter(self, K, *args,
                           criteria: str = 'RMSE', **kwargs,):

        if self.true_fn is None:
            raise ValueError("Can't compute best jitter without ground truth as reference")

        best_score = 1e10
        print('Searching for the best jitter value ...')
        for _jitter in [0.5, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14]:
            cholesky = K.cholesky(jitter = _jitter)
            pred, std = self.predict_cholesky(cholesky,
                                  *args, **kwargs)
            
            if criteria == 'RMSE':
                criteria_val = self.RMSE
            elif criteria == 'MAE':
                criteria_val = self.MAE
            else:
                criteria_val = self.L2RELATIVE

            if criteria_val < best_score:
                best_score = criteria_val
                bestjitter = _jitter
                bestprediction = pred
                beststd = std

        print(f'Best jitter found = {bestjitter} with {criteria} = {best_score}')
        self.prediction = bestprediction
        self.std_dev = beststd

        # saving the best jitter for future reference
        self._bestjitter = float(bestjitter)

        # recomputing the metrics
        self.metrics(self.prediction, self.true_fn)

        return self.prediction, self.std_dev

    # Full predict method -- needs minor modifications for each problem
    def predict(self, x_train, 
                x_test: Tensor,
                y_train: Tensor | None = None,
                z_train: Tensor | None = None,
                v_train: Tensor | None = None,
                ConcatedObs: Tensor | None = None, 
                centering: Tensor | float = 0.0,
                adaptive_nugget: bool = False,
                cholesky_mode: bool = True,
                best_jitter: bool = False,
                base_var: bool = True,                              # Making variance positive
                min_val: float = 0.0,                               # to clamp the conditional variance
                cross_covariance_fn: Callable | None = None,        # if predicting something else.
                sigma: float = 1.0,                                 # noise variance
                **kwargs,                                           # for the forward method AND cross-covariance
            ):

        # K matrix
        K = self.forward(x=x_train, 
                         z=z_train, 
                         adaptive_nugget=adaptive_nugget, 
                         **kwargs)
        
        # H matrix
        if cross_covariance_fn is None:
            cross_covariance = self.cross_covariance(x=x_train, 
                                                     x_test=x_test, 
                                                     z=z_train,
                                                     **kwargs)
        else:
            cross_covariance = cross_covariance_fn(x=x_train, 
                                                   x_test=x_test, 
                                                   z=z_train,
                                                   **kwargs)

        # Empty ConcatedObs
        if ConcatedObs is None and v_train is None:
            ConcatedObs = torch.cat([y_train,]).reshape(-1).double()
        elif ConcatedObs is None and v_train is not None:
            ConcatedObs = torch.cat([y_train, v_train]).reshape(-1).double()

        # # #======================
        # # EXAMPLE OF CENTERING IN THE MAIN CODE
        # # #======================
        # ConcatedObs, centering = CenteredValues(kernel = PhysicsRegression.kernel).ConcatenatedObs(y_train,
        #                                                                                            x_train, 
        #                                                                                            v_train,
        #                                                                                            z_train) 

        # Predicting in cholesky mode
        if cholesky_mode:
            if best_jitter:
                prediction, std_dev = self.search_best_jitter(K, 
                                                              cross_covariance, 
                                                              centering, 
                                                              ConcatedObs, x_test,
                                                              base_var,
                                                              min_val,
                                                              sigma,
                                                            )
            else:
                cholesky = K.cholesky(jitter = self.jitter)
                prediction, std_dev = self.predict_cholesky(cholesky, cross_covariance, 
                                                                centering,
                                                                ConcatedObs, x_test,
                                                                base_var,
                                                                min_val,
                                                                sigma)
        else:
            inverse = K.invert(nugget=self.nugget)
            prediction, std_dev = self.predict_inv(inverse, cross_covariance,
                                                   centering,
                                                   ConcatedObs, x_test,
                                                   base_var,
                                                   min_val,
                                                   sigma)

        # if self.save and best_jitter:
        #     self.save_model(directory_name=f'{self.__class__.__name__}',
        #                     jitter=self._bestjitter,
        #                     problem_name=f'{self.kernel.__class__.__name__}_bestjitt_{name}'
        #                 )

        # else:
        #     self.save_model(directory_name=f'{self.__class__.__name__}',
        #                     centered=ConcatedObs, centering=centering,
        #                     jitter=self.jitter,
        #                     Kmatrix = K, Hmatrix = cross_covariance,
        #                     problem_name=f'{self.kernel.__class__.__name__}_{name}'
        #                 )

        return prediction, std_dev    


    # LOOCV loss -- (pseudo - Dubrule's formula)
    def LOOCVloss(self, 
                  *args,                                            # for the forward function 
                  jitter = 1e-6, 
                  cholesky_mode = True,
                  ConcatedObs: Tensor | None = None,
                  y_train: Tensor | None = None, 
                  v_train: Tensor | None = None,    
                  filtering: list | None = None,
                  mode: str = 'normal',                             # OR 'sigma' 
                  **kwargs):                                        # for the forward function

        # Empty ConcatedObs
        if ConcatedObs is None and v_train is None:
            ConcatedObs = torch.cat([y_train,]).reshape(-1).double()
        elif ConcatedObs is None and v_train is not None:
            ConcatedObs = torch.cat([y_train, v_train]).reshape(-1).double()

        # K matrix
        K = self.forward(*args, **kwargs, diag_blocks=True)
        if cholesky_mode:
            inverse = torch.cholesky_inverse(K.cholesky(jitter=jitter))
        else: 
            inverse = K.invert(nugget = jitter)

        diag2 = (torch.diag(torch.diagonal(inverse)**(-2)))
        if mode == 'sigma':
            with torch.no_grad():
                diag2 = (torch.diag(torch.diagonal(inverse)**(-1)))             # Computing sigma

        # Diagonal blocks 
        diag_blocks = K.diag_blocks
        # filtering = torch.ones(len(diag_blocks)) if filtering is None else torch.tensor(filtering)
        filtering = torch.cat([torch.tensor([1.0]), torch.zeros(len(diag_blocks)-1)]) if filtering is None else torch.tensor(filtering)

        # sanity check
        if filtering.size(dim=0) is not len(diag_blocks):
            raise ValueError(f"Filter size {filtering.size(dim=0)} must match the number of diagonal blocks {len(diag_blocks)}")

        filter_matrices = []
        for diag, _ in zip(diag_blocks, range(len(diag_blocks))):
            entry = filtering[_] * torch.eye(diag.size(dim=0),
                                dtype = diag.dtype)
            filter_matrices.append(entry)

        filter_matrix = torch.block_diag(*filter_matrices)
        Ky = inverse @ ConcatedObs

        loss = (1 / (filtering * K.size_diags()).sum(dim=0)) * (Ky.unsqueeze(dim=-1).transpose(0, 1).squeeze(dim=-1) @ diag2 @ filter_matrix @ Ky)
        # loss = (1 / (filtering * K.size_diags()).sum(dim=0)) * (Ky.T @ diag2 @ filter_matrix @ Ky)        # to avoid warnings about .T, we use transpose(0, 1).

        if mode == 'sigma':
            with torch.no_grad():
                self.LOOCVsigma = loss.sqrt()
                return loss.sqrt() 
            
        return loss

    # Loss landscape visualization -- RBF, Matern type kernels.
    def loss_landscape_2D(self, 
                       *args,
                       txmin: float = 0.05,
                        txmax: float = 1.0,
                        tymin: float | None = None,
                        tymax: float | None = None,
                        Nx: int = 10,
                        Ny: int | None = None,
                        lossfn: str = 'LOOCV',
                        name: str = 'INSERT_FIG_NAME',
                       **kwargs):
        # create a list of kernel hyperparameters to check for.
        tx = torch.linspace(txmin, txmax, Nx)
        tymin = txmin if tymin is None else tymin
        tymax = txmax if tymax is None else tymax
        Ny = Nx if Ny is None else Ny
        ty = torch.linspace(tymin, tymax, Ny)

        # Compute the loss for each hyperparameter 
        lossvals = torch.zeros(Nx * Ny)

        # Storing the original lengthscale
        original_lengthscale = self.kernel.lengthscale
        original_diff_mode = self.kernel.diff_mode

        # lossfn
        if lossfn == 'LOOCV':
            _loss = self.LOOCVloss
        elif lossfn == 'NLL':
            _loss = self.NLLloss

        _ = 0                       # counter
        best_val = 1e10
        for _i in tx:   
            for _j in ty: 
                # Kernel(s) to use (only for RBFKernel type kernels)
                self.kernel.lengthscale = torch.tensor([_i, _j])
                self.kernel.diff_mode = True

                lossvals[_] = _loss(*args, **kwargs)

                with torch.no_grad():
                    if lossvals[_] < best_val:
                        best_val = lossvals[_]
                        besttx = _i
                        bestty = _j 
                    
                    _+=1

        self.kernel.lengthscale = original_lengthscale
        self.kernel.diff_mode = original_diff_mode

        # Visualize
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(projection='3d')
        
        t1, t2 = torch.meshgrid(tx, ty, indexing = 'xy')
        
        surface = ax.plot_surface(t1.cpu(), t2.cpu(), 
                                  lossvals.reshape(Nx, Ny).T.detach().numpy(), 
                                  cmap='inferno', edgecolor='gray')

        ax.set_zlim(0.0,
                    torch.quantile(lossvals, 0.75, interpolation='nearest').item()
                )

        ax.set_title('loss landscape', fontsize=14)
        ax.set_xlabel(r'$\theta_1$', fontsize=10)
        ax.set_ylabel(r'$\theta_2$', fontsize=10)
        ax.set_zlabel('loss', fontsize=10)

        # Adjust the viewing perspective if needed
        ax.view_init(elev=40, azim=30)
        ax.xaxis.set_inverted(True)
        ax.yaxis.set_inverted(False)

        ax.scatter(besttx, bestty, best_val.item(), 
                   color='red', s=100, marker = '*')
        print(f'Best lengthscale from loss landscape exploration: {[besttx.item(), bestty.item()]} with loss: {best_val.item()}')
        
        fig.colorbar(surface, shrink=0.5, aspect=10)

        if self.save:
            self.best_landscape_loss = torch.tensor([besttx.item(), bestty.item()])
            self.save_fig(directory_name=f'{self.__class__.__name__}Figures',
                            fig_name=f'Loss_landscape_2D_{self.kernel.lengthscale.tolist()}_{name}'
                        )
        
        plt.show()

    # Visualization of the results
    def visualize_2D(self, x_train: Tensor,
                  x_test: Tensor,
                  y_train: Tensor | None = None,
                  option: str = 'comparison', 
                  limit: bool = False,
                name: str = 'INSERT_FIG_NAME',
                  **kwargs):

        x_train_plot = x_train.detach().numpy()
        y_train_plot = y_train.detach().numpy()
        x_test_plot = x_test.detach().numpy()

        prediction_plot = self.prediction.detach().numpy()
        true_fn_plot = self.true_fn.detach().numpy()
        std_plot = self.std_dev.detach().numpy()

        # To plot the comparison between true and predicted.
        if option == 'comparison':
            figure, axes = plt.subplots(1, 3, figsize=(20, 4.5)) 
            
            shape = self.true_fn.size()

            yMAX = torch.max(self.true_fn)                    
            yMIN = torch.min(self.true_fn)

            # 1. Regression results
            cm = axes[0].contourf(x_test_plot[:,0].reshape(*shape), 
                                  x_test_plot[:,1].reshape(*shape), 
                                  prediction_plot.reshape(*shape), 
                                  **kwargs)
            axes[0].scatter(x_train_plot[:, 0], x_train_plot[:, 1], marker = 'X', s=10, c='black')
            axes[0].set_title(f'CK {self.kernel} prediction')
            if limit:
                cm.set_clim(yMIN, yMAX)                                     # Range of true solution
            axes[0].set_xlim(min(x_test_plot[:,0]),
                             max(x_test_plot[:,0]))
            axes[0].set_ylim(min(x_test_plot[:,1]),
                            max(x_test_plot[:,1]))
            figure.colorbar(cm)
            
            # 2. True function
            cm = axes[1].contourf(x_test_plot[:,0].reshape(*shape), 
                                  x_test_plot[:,1].reshape(*shape), true_fn_plot.reshape(*shape), 
                                  **kwargs)
            axes[1].set_title(r'True solution')
            if limit:
                cm.set_clim(yMIN, yMAX)                                     # Range of true solution
            figure.colorbar(cm)
            
            # 3. UQ
            cm = axes[2].contourf(x_test_plot[:,0].reshape(*shape), 
                                  x_test_plot[:,1].reshape(*shape), std_plot.reshape(*shape),
                                  cmap = 'inferno', levels = 100)
            axes[2].set_title(r'$\pm 2 std.$')
            figure.colorbar(cm)

            if self.save:
                self.save_fig(f'{self.__class__.__name__}Figures',
                    f'CK_{self.kernel.__class__.__name__}_{self.kernel.lengthscale.tolist()}_{name}')
            
            if not self.save:
                plt.show()

        elif option == 'CK':
            figure, axes = plt.subplots(1, 1, figsize=(9, 5))

            shape = self.true_fn.size()
            
            yMAX = torch.max(self.true_fn)                    
            yMIN = torch.min(self.true_fn)

            # 1. Regression results
            cm = axes.contourf(x_test_plot[:,0].reshape(*shape), 
                                  x_test_plot[:,1].reshape(*shape), 
                                  prediction_plot.reshape(*shape), 
                                  **kwargs)
            axes.scatter(x_train_plot[:, 0], x_train_plot[:, 1], marker = 'X', s=10, c='black')
            axes.set_title(f'CK {self.kernel} prediction')
            if limit:
                cm.set_clim(yMIN, yMAX)                                     # Range of true solution
            axes.set_xlim(min(x_test_plot[:,0]),
                             max(x_test_plot[:,0]))
            axes.set_ylim(min(x_test_plot[:,1]),
                            max(x_test_plot[:,1]))
            figure.colorbar(cm)

            if self.save:
                self.save_fig(f'{self.__class__.__name__}Figures',
                    f'OnlyCK_{self.kernel.__class__.__name__}_{self.kernel.lengthscale.tolist()}_{name}')
            if not self.save:
                plt.show()
        
        elif option == 'SK':
            figure, axes = plt.subplots(1, 1, figsize=(9, 5))
            
            shape = self.true_fn.size()
            
            yMAX = torch.max(self.true_fn)                    
            yMIN = torch.min(self.true_fn)
            
            # 1. Regression results
            cm = axes.contourf(x_test_plot[:,0].reshape(*shape), 
                                  x_test_plot[:,1].reshape(*shape), 
                                  self.predictionSK.cpu().reshape(*shape), 
                                  **kwargs)
            axes.scatter(x_train_plot[:, 0], x_train_plot[:, 1], marker = 'X', s=10, c='black')
            axes.set_title(f'SK {self.sk_kernel} prediction')
            if limit:
                cm.set_clim(yMIN, yMAX)                                     # Range of true solution
            axes.set_xlim(min(x_test_plot[:,0]),
                             max(x_test_plot[:,0]))
            axes.set_ylim(min(x_test_plot[:,1]),
                            max(x_test_plot[:,1]))
            figure.colorbar(cm)

            if self.save:
                self.save_fig(f'{self.__class__.__name__}Figures',
                    f'SK_{self.kernel.__class__.__name__}_{name}')

            if not self.save:
                plt.show()

        # To plot the true.
        if option == 'true':
            figure, axes = plt.subplots(1, 1, figsize=(9, 5)) 
            
            shape = self.true_fn.size()
        
            yMAX = torch.max(self.true_fn)                    
            yMIN = torch.min(self.true_fn)

            cm = axes.contourf(x_test_plot[:,0].reshape(*shape), 
                                  x_test_plot[:,1].reshape(*shape), true_fn_plot.reshape(*shape), 
                                  **kwargs)
            axes.set_title(r'True soln.')
            if limit:
                cm.set_clim(yMIN, yMAX)                                     # Range of true solution
            figure.colorbar(cm)

            if self.save:
                self.save_fig(f'{self.__class__.__name__}Figures',
                    f'Truth_{name}')
            
            if not self.save:
                plt.show()
        # To plot the UQ.
        if option == 'UQ':
            figure, axes = plt.subplots(1, 1, figsize=(9, 5)) 
            
            shape = self.true_fn.size()
        
            yMAX = torch.max(self.true_fn)                    
            yMIN = torch.min(self.true_fn)
        
            cm = axes.contourf(x_test_plot[:,0].reshape(*shape), 
                                  x_test_plot[:,1].reshape(*shape), std_plot.reshape(*shape), 
                                  cmap = 'inferno', levels = 100)
            axes.set_title(r'$\pm 2 std.$')
            if limit:
                cm.set_clim(yMIN, yMAX)                                     # Range of true solution
            figure.colorbar(cm)
        
            if self.save:
                self.save_fig(f'{self.__class__.__name__}Figures',
                    f'UQ_{self.kernel.__class__.__name__}_{self.kernel.lengthscale.tolist()}_{name}')
            
            if not self.save:
                plt.show()
    # Visualization of the results
    def visualize_1D(self, x_train: Tensor,
                     x_test: Tensor,
                     y_train: Tensor | None = None,  
                     option: str = 'comparison',
                     limit: bool = False, 
                    name: str = 'INSERT_FIG_NAME',
                     **kwargs):
        
        x_train_plot = x_train.detach().numpy().squeeze()
        y_train_plot = y_train.detach().numpy().squeeze()
        x_test_plot = x_test.detach().numpy().squeeze()
        
        prediction_plot = self.prediction.detach().numpy().squeeze()
        true_fn_plot = self.true_fn.detach().numpy().squeeze()
        std_plot = self.std_dev.detach().numpy().squeeze()
        
        # To plot the comparison between true and predicted.
        if option == 'CK':
            figure, axes = plt.subplots(1, 1, figsize=(10, 7)) 
            
            yMAX = torch.max(self.true_fn)                    
            yMIN = torch.min(self.true_fn)
        
            # Regression results
            axes.plot(x_test_plot,  
                    prediction_plot, 
                    c = 'blue', linestyle = 'dashed', 
                    **kwargs)
            # UQ.
            axes.fill_between(x_test_plot, 
                                 prediction_plot - 2.0 * std_plot,
                                 prediction_plot + 2.0 * std_plot,
                                 alpha = 0.5, color = 'blue', **kwargs)
            # True function
            axes.plot(x_test_plot, true_fn_plot,
                         c = 'black', **kwargs)
            axes.scatter(x_train_plot, y_train_plot, marker = 'X', s=50, c='black')
            axes.set_title(f'CK {self.kernel} prediction')
            if limit:
                axes.set_ylim(yMIN - 0.25, yMAX + 0.25)                                     # Range of true solution

            if self.save:
                self.save_fig(f'{self.__class__.__name__}Figures',
                    f'OnlyCK_1D_{self.kernel.__class__.__name__}_{self.kernel.lengthscale.tolist()}_{name}')

            plt.show()

        elif option == 'SK':
            figure, axes = plt.subplots(1, 1, figsize=(10, 7)) 
            
            yMAX = torch.max(self.true_fn)                    
            yMIN = torch.min(self.true_fn)

            # Simple Kriging
            axes.plot(x_test_plot,  
                        self.predictionSK.cpu(), 
                         c = 'red', linestyle = 'dashed', 
                        **kwargs)
            
            # UQ.
            axes.fill_between(x_test_plot, 
                                 self.lowerSK.cpu(),
                                 self.upperSK.cpu(),
                                 alpha = 0.5, color = 'red', **kwargs)
            # True function
            axes.plot(x_test_plot, true_fn_plot,
                         c = 'black', **kwargs)
            axes.scatter(x_train_plot, y_train_plot, marker = 'X', s=50, c='black')
            axes.set_title(f'SK {self.sk_kernel} prediction')
            if limit:
                axes.set_ylim(yMIN - 0.1, yMAX + 0.1)                                     # Range of true solution

            if self.save:
                self.save_fig(f'{self.__class__.__name__}Figures',
                    f'SK_1D_{self.kernel.__class__.__name__}_{self.kernel.lengthscale.tolist()}_{name}')
            
            plt.show()

        elif option == 'comparison':
            figure, axes = plt.subplots(1, 1, figsize=(10, 7)) 
            
            yMAX = torch.max(self.true_fn)                    
            yMIN = torch.min(self.true_fn)
            
            # Regression results
            axes.plot(x_test_plot,  
                        prediction_plot, 
                        c = 'blue', linestyle = 'dashed', 
                        **kwargs)

            axes.plot(x_test_plot,  
                        self.predictionSK.cpu(), 
                         c = 'red', linestyle = 'dashed', 
                        **kwargs)
            
            # UQ.
            axes.fill_between(x_test_plot, 
                                 prediction_plot - 2.0 * std_plot,
                                 prediction_plot + 2.0 * std_plot,
                                 alpha = 0.5, color = 'blue', **kwargs)
            axes.fill_between(x_test_plot, 
                                 self.lowerSK.cpu(),
                                 self.upperSK.cpu(),
                                 alpha = 0.5, color = 'red', **kwargs)
            # True function
            axes.plot(x_test_plot, true_fn_plot,
                         c = 'black', **kwargs)
            axes.scatter(x_train_plot, y_train_plot, marker = 'X', s=50, c='black')
            axes.set_title(f'CK {self.kernel} + SK {self.sk_kernel} prediction')
            if limit:
                axes.set_ylim(yMIN - 0.1, yMAX + 0.1)                                     # Range of true solution

            if self.save:
                self.save_fig(f'{self.__class__.__name__}Figures',
                    f'CK_1D_{self.kernel.__class__.__name__}_{self.kernel.lengthscale.tolist()}_{name}')

            plt.show()

    def save_model(self, directory_name: str = r'CoKrigingModels', 
                   jitter = 1e-6, 
                   prediction: Tensor | None = None,
                   std_dev: Tensor | None = None, 
                   problem_name: str = 'Something',
                ):
        """ save the co-Kriging object """
        if self.save:
            # directory_name = r'CoKrigingModels'
            with torch.no_grad():
                dictionary = dict(
                    lengthscale = self.kernel.lengthscale.detach(),
                    best_landscape_loss = self.best_landscape_loss.detach() if hasattr(self, 'best_landscape_loss') else None, 
                    jitter = float(jitter),             # pass the best jitter if possible
                    RMSE = self.RMSE.detach(),
                    MAE = self.MAE.detach(),
                    L2RELATIVE = self.L2RELATIVE.detach(),
                    LOOCVsigma = self.LOOCVsigma.detach() if hasattr(self, 'LOOCVsigma') else None,
                    RMSE_SK = self.RMSE_SK.detach() if hasattr(self, 'RMSE_SK') else None,
                    MAE_SK = self.MAE_SK.detach() if hasattr(self, 'MAE_SK') else None,
                    L2RELATIVE_SK = self.L2RELATIVE_SK.detach() if hasattr(self, 'L2RELATIVE_SK') else None,
                    beta = self.beta if hasattr(self, 'beta') else None, 
                    rho = self.rho if hasattr(self, 'rho') else None,
                    nu = self.nu if hasattr(self, 'nu') else None
                )

            try:
                os.mkdir(directory_name)
                print(f"Directory '{directory_name}' created successfully.")
                torch.save(dictionary, f'{directory_name}/{problem_name}.pt')
                # _file = open(f'{directory_name}/{problem_name}.json', 'wb')
                # pickle.dump(dictionary, _file)
                # _file.close()

            except FileExistsError:
                print(f"Directory '{directory_name}' already exists.")
                torch.save(dictionary, f'{directory_name}/{problem_name}.pt')
                # _file = open(f'{directory_name}/{problem_name}', 'wb')
                # pickle.dump(dictionary, _file)
                # _file.close()
            except PermissionError:
                print(f"Permission denied: Unable to create '{directory_name}'.")
            except:
                print("Something went wrong while saving model")

            # if save_matrices and Kmatrix is None and Hmatrix is None:
            #     self.Kmatrix = self.forward(*args, **kwargs)
            #     self.Hmatrix = self.cross_covariance(*args, **kwargs)
            # elif save_matrices and Kmatrix is not None and Hmatrix is not None:
            #     self.Kmatrix = Kmatrix
            #     self.Hmatrix = Hmatrix

            # self.centered = centered                # necessarily pass the centered
            # self.centering = centering              # necessarily pass the centering

        else:
            raise ValueError('Warning: trying to save co-Kriging model')

    def save_fig(self, directory_name: str = r'CoKrigingFigures', 
                 fig_name: str = 'Something'):
        """ save the figure """
        if self.save:
            # directory_name = r'CoKrigingFigures'

            # Create the directory
            try:
                os.mkdir(directory_name)
                print(f"Directory '{directory_name}' created successfully.")
                plt.savefig(f'{directory_name}/{fig_name}.pdf', bbox_inches = 'tight', pad_inches = 0)
            except FileExistsError:
                plt.savefig(f'{directory_name}/{fig_name}.pdf', bbox_inches = 'tight', pad_inches = 0)
            except PermissionError:
                print(f"Permission denied: Unable to create '{directory_name}'.")
            except Exception as e:
                print(f"An error occurred: {e}") 

        else:
            raise ValueError('Warning: trying to save FIGURE')
#===================================
# A generic training mode -- as a function of the PhysicsRegressor class.
# Optional class, could have also just defined a function I think ?
#===================================
class fit_regressor():

    """A finite differences implementation of the general_fit_regressor(). torch based optimizer TODO"""

    def __init__(
        self,
        model: PhysicsRegression,                               # the co-Kriging model
        lossfn: str = 'LOOCV',
        gpy_mode: bool = False,                                 # gpytorch optimizer                                      
        steps: int = 500,
        learning_rate: float = 0.01,
        interactive: bool = False,
    ) -> None:

        super().__init__()

        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")

        self.model = model

        if not gpy_mode:
            if lossfn == 'LOOCV':
                self.lossfn = self.model.LOOCVloss
            elif lossfn == 'NLL':
                self.lossfn = self.model.NLLloss

        self.steps = int(steps)
        self.lr = float(learning_rate)

        self.interactive = bool(interactive)
        self.gpy_mode = bool(gpy_mode)


    # Finite differences based implementation
    def central_differences_train(self, *args,                                                  # for the loss function
                                  gradient_eps = 1e-4,
                                  find_best = True,                                             # Output the best lengthscale -- if False, returns the last lengthscale.
                                  weight_decay = 1e-4,                                          # for the optimizer
                                  **kwargs):                                                    # for the loss function

        if self.gpy_mode is True:
            raise ValueError('Use `gpy_train()` for gpytorch based models')

        lengthscales: list[Tensor] = []
        gradients: list[Tensor] = []
        losses: list[float] = []
        
        if list(self.model.parameters()) == []:
            print('Model has no parameters to optimize.')                   # LinearKernel(), for example.
            losses = self.steps*[self.lossfn(*args, **kwargs)]
            return losses
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, 
                                     weight_decay=weight_decay,
                            )
        # optimizer = torch.optim.SGD(self.model.parameters(), lr = self.lr,
        #                             # weight_decay=1e-4,
        #                     )

        # announce co-Kriging training
        _CK_flash()
        self.model.train()
        
        # Saving the best hyperparameters
        best = None

        loss = self.lossfn(*args, **kwargs)
        print(f'Pre-training loss: {loss.detach().item()}')

        for _, _num  in zip(
            tqdm(range(self.steps), colour = 'red', position=0),
            range(self.steps)
        ):
            optimizer.zero_grad(set_to_none=True)
            loss = self.lossfn(*args, **kwargs)
        
            # compulsory pass in the loop
            if _num == 0:
                best = loss.item()                              # Initiate the best value
                bestparam = self.model.kernel.lengthscale       # Initiate the lengthscale
        
            # optional pass: if loss is lower, best = loss.
            if best >= loss.item():
                bestparam = self.model.kernel.lengthscale       # Saving the best lengthscale
                best = loss.item()                              # New best score
        
            if self.interactive:
                print(f'{self.model.kernel.lengthscale.tolist()} loss at step {_num}/{self.steps} - {loss.item():.2f}',) 
                # end='\r', flush=True)
        
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss encountered during training")
        
            for param in self.model.parameters():

            #===================================
            # Central differences implementation
            #===================================
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
                  p_data[idx] = original_val + gradient_eps
                  loss_high = self.lossfn(*args, **kwargs).item()
                  if self.interactive:
                    print(f'loss_high = {loss_high} for parameter = {(original_val + gradient_eps):.4f}')
                
                  # 2. Evaluate loss at x - eps
                  p_data[idx] = original_val - gradient_eps
                  loss_low = self.lossfn(*args, **kwargs).item()
                  if self.interactive:
                    print(f'loss_low = {loss_low} for parameter = {(original_val - gradient_eps):.4f}')
        
                  # modify p_grad
                  p_grad[idx] = ((loss_high - loss_low) / (2 * gradient_eps))
        
                  gradlist.append(((loss_high - loss_low) / (2 * gradient_eps)))
            
                  p_data[idx] = original_val
                if self.interactive:
                    print(f'Central differences gradient approximate: {gradlist}')
        
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            optimizer.step()
            losses.append(loss.detach().item())
        
        if  not find_best or self.steps == 0:
            print(f'Final lengthscale: {self.model.kernel.lengthscale.tolist()} and LOOCV score {loss.detach().item()}')
        else:  
            print(f'Best lengthscale {bestparam.tolist()} and best LOOCV score: {best}')
            self.model.kernel.lengthscale = bestparam 

        self.model.eval()
        self.losses = losses

    # Plotting the evolution of the loss function over training iterations.
    def plot_loss(self,):
        if not self.steps == 0:
            plt.figure(figsize=(10,12))
            plt.plot(range(self.steps), self.losses, 'b--')
            plt.xlabel('iterations')
            plt.ylabel('loss function Co-Kriging')
            
            if not self.model.save:
                plt.show()

    # gpytorch trainer.
    def gpy_train(self, 
                  *args,
                  loss: MyLeaveOneOutPseudoLikelihood = MyLeaveOneOutPseudoLikelihood,
                  kernel_cls = gpytorch.kernels.RBFKernel,
                  **kwargs,):

        self.sk_kernel = kernel_cls

        losses, mean, (lower, upper) = gpytorch_regressor(*args,
                                    _loss = loss,
                                    lr = self.lr,
                                    _jitter = self.model.jitter,
                                    u_true = self.model.true_fn,
                                    iter = self.steps,
                                    interactive = self.interactive,
                                    kernel_cls= self.sk_kernel,
                                    **kwargs,
                                    )
        
        # for the fit_regressor class
        self.lossesSK = losses
        self.predictionSK = mean
        self.lowerSK = lower
        self.upperSK = upper 

    # pass fit_regressor predictions to model class
    def pass_to_model(self, mean, lower, upper):

        # for the CoKrig class
        self.model.predictionSK = mean
        self.model.lowerSK = lower
        self.model.upperSK = upper

        self.model.sk_kernel = self.sk_kernel  

    # Computing the error metrics
    def metrics_SK(self, prediction, 
                   reference, rel_tol: float | None = None) -> None:

        rmse = torch.mean((prediction.reshape(*reference.size()) - reference)**2)**0.5
        mae = torch.mean((torch.abs(prediction.reshape(*reference.size()) - reference)))
    
        # rel_tol = torch.mean(reference) if rel_tol is None else rel_tol
        rel_tol = 1e-2 if rel_tol is None else rel_tol
        rel_tol = 1e-2 if rel_tol <= 1e-2 else rel_tol

        l2relative = torch.linalg.norm((prediction.reshape(*reference.size()) - reference).flatten(), 
                                       ord = 2, dim = 0) / torch.linalg.norm(reference.flatten(), 
                                                                             ord = 2, dim = 0)  

        # l2relative = torch.mean((
        #     torch.abs(prediction.reshape(*reference.size()) - reference) / (
        #         torch.abs(reference) + rel_tol)
        #         ))

        # Error attributes (for PhysicsRegression class)
        self.model.RMSE_SK = rmse
        self.model.MAE_SK = mae
        self.model.L2RELATIVE_SK = l2relative

    # Plotting the evolution of the loss function over training iterations.
    def plot_loss_SK(self,):
        if not self.steps == 0:
            plt.figure(figsize=(10,12))
            plt.plot(range(self.steps), self.lossesSK, 'r--')
            plt.xlabel('iterations')
            plt.ylabel('loss function Simple Kriging')
            plt.show()

    
        

    





        


    
    
                



