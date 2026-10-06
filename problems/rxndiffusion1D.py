# The PINN failure case taken from https://arxiv.org/pdf/2109.01050: Non-linear case

# imports
import sys
import torch
import numpy as np
import math 

sys.path.append('..')           # To search in the right directory (git-PICOKRIG)

from kernels import expanding_inputs, Tensor, kernel_derivatives_2D, symmetry_check, printRed, dim_check, fit_regressor 

from physics import PhysicsRegression, PhysicsMatrix, _adaptive_nugget_constructor, _symmetric_block, CenteredValues, fit_regressor
from kernels import kernel_derivatives_2D, expanding_inputs, printRed, symmetry_check, dim_check, IsserlisPow12, IsserlisPow22, choose_kernel
from SimpleKrigGpytorch import choose_gpy_kernel
from torch import Tensor
import numpy as np
import math
from typing import Literal
#==================
# Design of experiments
#==================

#******************
# Taken from Krishnapayan et al.: https://github.com/a1k12/characterizing-pinns-failure-modes/blob/main/pbc_examples/systems_pbc.py
#******************
def function(u0: str):
    """Initial condition, string --> function."""

    if u0 == 'sin(x)':
        u0 = lambda x: np.sin(x)
    elif u0 == 'sin(pix)':
        u0 = lambda x: np.sin(np.pi*x)
    elif u0 == 'sin^2(x)':
        u0 = lambda x: np.sin(x)**2
    elif u0 == 'sin(x)cos(x)':
        u0 = lambda x: np.sin(x)*np.cos(x)
    elif u0 == '0.1sin(x)':
        u0 = lambda x: 0.1*np.sin(x)
    elif u0 == '0.5sin(x)':
        u0 = lambda x: 0.5*np.sin(x)
    elif u0 == '10sin(x)':
        u0 = lambda x: 10*np.sin(x)
    elif u0 == '50sin(x)':
        u0 = lambda x: 50*np.sin(x)
    elif u0 == '1+sin(x)':
        u0 = lambda x: 1 + np.sin(x)
    elif u0 == '2+sin(x)':
        u0 = lambda x: 2 + np.sin(x)
    elif u0 == '6+sin(x)':
        u0 = lambda x: 6 + np.sin(x)
    elif u0 == '10+sin(x)':
        u0 = lambda x: 10 + np.sin(x)
    elif u0 == 'sin(2x)':
        u0 = lambda x: np.sin(2*x)
    elif u0 == 'tanh(x)':
        u0 = lambda x: np.tanh(x)
    elif u0 == '2x':
        u0 = lambda x: 2*x
    elif u0 == 'x^2':
        u0 = lambda x: x**2
    elif u0 == 'gauss':
        x0 = np.pi
        sigma = np.pi/4
        u0 = lambda x: np.exp(-np.power((x - x0)/sigma, 2.)/2.)
    return u0

def reaction(u, rho, dt):
    """ du/dt = rho*u*(1-u)
    """
    factor_1 = u * np.exp(rho * dt)
    factor_2 = (1 - u)
    u = factor_1 / (factor_2 + factor_1)
    return u

def diffusion(u, nu, dt, IKX2):
    """ du/dt = nu*d2u/dx2
    """
    factor = np.exp(nu * IKX2 * dt)
    u_hat = np.fft.fft(u)
    u_hat *= factor
    u = np.real(np.fft.ifft(u_hat))
    return u

def reaction_solution(u0: str, rho, nx=256, nt=100):
    L = 2*np.pi
    T = 1
    dx = L/nx
    dt = T/nt
    x = np.arange(0, 2*np.pi, dx)
    t = np.linspace(0, T, nt).reshape(-1, 1)
    X, T = np.meshgrid(x, t)

    # call u0 this way so array is (n, ), so each row of u should also be (n, )
    u0 = function(u0)
    u0 = u0(x)

    u = reaction(u0, rho, T)

    u = u.flatten()
    return u

def reaction_diffusion_discrete_solution(u0 : str, nu, rho, nx = 256, nt = 100):
    """ Computes the discrete solution of the reaction-diffusion PDE using
        pseudo-spectral operator splitting.
    Args:
        u0: initial condition
        nu: diffusion coefficient
        rho: reaction coefficient
        nx: size of x-tgrid
        nt: number of points in the t grid
    Returns:
        u: solution
    """
    L = 2*np.pi
    T = 1
    dx = L/nx
    dt = T/nt
    x = np.arange(0, L, dx) # not inclusive of the last point
    t = np.linspace(0, T, nt).reshape(-1, 1)
    X, T = np.meshgrid(x, t)
    u = np.zeros((nx, nt))

    IKX_pos = 1j * np.arange(0, nx/2+1, 1)
    IKX_neg = 1j * np.arange(-nx/2+1, 0, 1)
    IKX = np.concatenate((IKX_pos, IKX_neg))
    IKX2 = IKX * IKX

    # call u0 this way so array is (n, ), so each row of u should also be (n, )
    u0 = function(u0)
    u0 = u0(x)

    u[:,0] = u0
    u_ = u0
    for i in range(nt-1):
        u_ = reaction(u_, rho, dt)
        u_ = diffusion(u_, nu, dt, IKX2)
        u[:,i+1] = u_

    u = u.T
    u = u.flatten()
    return u

# The design of experiments
def rxndiffusion_DoE(nu: float = 5, rho: float = 5, N_initial: int = 60,
                     Nx: int  = 256, Nt: int  = 100, N_boundary: int = 60,
                     N_rand: int = 0, N_colloc: int = 30,
                     random_colloc: bool = False, 
                     seed: int = 0, t_obs: float | None = None,
                     u0: str = 'gauss', 
                     **kwargs):

    generator = torch.Generator().manual_seed(seed)

    # Calling the solution function
    u_true = reaction_diffusion_discrete_solution(u0 = u0, nu = nu, rho = rho,
                                                nx = Nx, nt = Nt).reshape(Nt, Nx)

    u0 = function(u0)
    
    X_min = 0.0; X_max = 2*np.pi
    T_min = 0.0; T_max = 1.0

    dx = (X_max - X_min)/Nx

    x = np.arange(X_min, X_max, dx) # not inclusive of the last point

    t = np.linspace(T_min, T_max, Nt).reshape(-1, 1)
    X, T = np.meshgrid(x, t)

    # # Visualize
    # plt.figure(figsize = (20,10))
    # plt.contourf(T, X, u_true, cmap = 'rainbow', levels =100)
    # plt.title('True function')
    # plt.colorbar()
    # plt.show()

    u_true = torch.tensor(u_true).double()
    xgrid = torch.arange(X_min, X_max, dx)
    tgrid = torch.linspace(T_min, T_max, Nt)

    # # Check size
    # print(u_true.size())

    #====================
    # Random observations in the domain
    #====================
    
    # Random indices
    repeat = int(N_rand / Nx)               # repeat the indices if N_rand is bigger than the original number of indices
    xindices = torch.randperm(Nx, generator=generator).tolist()
    for repition in range(repeat):
        xtemp = torch.randperm(Nx, generator=generator).tolist()
        xindices.extend(xtemp)

    repeat = int(N_rand / Nt)               # repeat the indices if N_rand is bigger than the original number of indices
    tindices = torch.randperm(Nt, generator=generator).tolist()
    for repition in range(repeat):
        ttemp = torch.randperm(Nt, generator=generator).tolist()
        tindices.extend(ttemp)

    xidx = xindices[:N_rand]
    tidx = tindices[:N_rand]

    Random = torch.stack([tgrid[tidx], xgrid[xidx]],
                          dim = 1)
    y_Random = u_true[tidx, xidx]

    # Visualize, set N_rand  = 2000
    # plt.scatter(Random[:,0].cpu(), Random[:,1].cpu(), c = y_Random, cmap = 'rainbow')
    # plt.show()


    #====================
    # Initial condition as observations
    #====================
    # All points at t = 0 
    X_range = torch.linspace(X_min, X_max, N_initial)
    Initial = torch.stack([torch.zeros(N_initial), 
                           X_range],
                           dim = 1)
    
    u_initial = u0(X_range.numpy())
    y_Initial = torch.tensor(u_initial).double()
    # print(f'{Initial=}')
    
    # mu = math.pi
    # sigma = math.pi / 4.0
    # y_Initial = torch.exp(-((Initial[:,1] - mu) / sigma)**2 / 2.0)          # u(0, x) ~ N(mu, sigma)

    # # Visualize
    # plt.plot(X_range.cpu(), y_Initial.cpu())
    # plt.show()

    #====================
    # Periodic boundary conditions
    #====================
    # u(0,t) = u(2pi,t) 
    T_range = torch.linspace(T_min, T_max, N_boundary)
    BoundaryUpper = torch.stack([T_range,
                                 X_max * torch.ones(N_boundary)],
                                 dim = 1)
    BoundaryLower = torch.stack([T_range,
                                 X_min * torch.ones(N_boundary)],
                                 dim = 1)
    # print(f'{BoundaryUpper=}')
    # print(f'{BoundaryLower=}')
    # # Visualize
    # plt.scatter(BoundaryLower[:,0].cpu(), BoundaryLower[:,1].cpu(), marker = 'X', c = 'green')
    # plt.scatter(BoundaryUpper[:,0].cpu(), BoundaryUpper[:,1].cpu(), marker = 'X', c = 'red')
    # plt.title('Boundary diagnosis')
    # plt.show()

    #====================
    # Test points
    #====================
    # Evaluating the model.  
    x_test = torch.stack(
            torch.meshgrid(tgrid, xgrid, indexing='ij'),
            dim=-1
        ).reshape(Nt * Nx, 2)

    # # Visualize
    # plt.scatter(x_test[:,0].cpu(), x_test[:,1].cpu(), marker = 'X', c = range(x_test.size(dim = 0)), cmap = 'rainbow')
    # plt.colorbar()
    # plt.title('Test points diagnosis')
    # plt.show()
    
    #====================
    # Collocation points
    #====================
    if not random_colloc:
        if 'N_colloc_t_disc' in kwargs and 'N_colloc_x_disc' in kwargs:
        # Inequal grids
            N_colloc_t_disc = kwargs.get('N_colloc_t_disc')
            N_colloc_x_disc = kwargs.get('N_colloc_x_disc')
        
        else: 
        # Equal grids
            disc = int(math.sqrt(N_colloc))
            N_colloc_t_disc = disc
            N_colloc_x_disc = disc
        
        T_range = torch.linspace(T_min, T_max, N_colloc_t_disc)
        X_range = torch.linspace(X_min, X_max, N_colloc_x_disc)
        Colloc = torch.stack(
                    torch.meshgrid(T_range, X_range, indexing='xy'),
                    dim=-1
                ).reshape(N_colloc_t_disc * N_colloc_x_disc, 2)
    
    # Random collocation points
    elif random_colloc:
        raw_colloc = torch.rand(N_colloc, 2, generator=generator)
        
        # Proper scaling
        Colloc = torch.stack([T_min + (T_max - T_min) * raw_colloc[:,0],
                            X_min + (X_max - X_min) * raw_colloc[:,1]],
                            dim = 1)
    
    y_colloc = torch.zeros(Colloc.size(dim=0))                          # Homogeneous RHS
    
    # # Visualize
    # plt.scatter(Colloc[:,0].cpu(), Colloc[:,1].cpu(), marker = 'X')
    # plt.title('Collocation diagnosis')
    # plt.show()
    
    x_train = torch.cat([Random,
                        Initial,])
    
    # Stacked observation values
    y_train = torch.cat([y_Random,
                        y_Initial,])

    # plt.figure(figsize = (20,10))
    # plt.contourf(x_test[:,0].reshape(*u_true.size()).cpu(), 
    #              x_test[:,1].reshape(*u_true.size()).cpu(), 
    #              u_true.cpu(), cmap = 'rainbow', levels = 100)
    # plt.colorbar()
    # plt.title('True function')
    # plt.show()
    
    return u_true, x_train.double(), BoundaryLower.double(), BoundaryUpper.double(), Colloc.double(), x_test.double(), y_train.double(), y_colloc.double()


#===================
# Physics kernel regressor
#===================
class RxnDiffusionRegressor(PhysicsRegression):
    r"""
    Reaction diffusion equation co-Kriging class. 
    
    Define the following methods:
        1. `.forward()`: this is going to give the observation covariance matrix.
        2. `.cross_covariance(mode:str='u')`: define one or multiple modes. 
    """

    def __init__(
            self,
            *args,
            rho: float = 5.0,
            nu: float = 5.0,
            mean: float = 0.0,
            mode: str = 'u2',
            boundary_mode: Literal['BL2 - BU2', '(BL - BU)^2'] = 'BL2 - BU2',       
            **kwargs,
    ) -> None:

        # Passing to the parent class 
        super().__init__(*args, nu = nu, rho = rho, mode = mode, **kwargs)          # rho = rho 

        self.mean = float(mean)
        self.boundary = str(boundary_mode)

    def IsserlisPow12(self, *args, **kwargs):
        """ Alias for IsserlisPow12 """
        return IsserlisPow12(*args, mean  = self.mean , **kwargs,)

    def IsserlisPow22(self, *args, **kwargs):
        """ Alias for IsserlisPow22 """
        return IsserlisPow22(*args, mean  = self.mean , **kwargs,)

    def CovUC(self, K, Kt, Kxx):
        """ Covariance between observations and collocation points """
        return Kt - (self.nu * Kxx) - (self.rho * K) + (self.rho * self.IsserlisPow12(K))

    def CovU2C(self, K, Kt, Kxx):
        """ Covariance between observations of u^2 and collocation points """
        return self.IsserlisPow12(Kt) - (self.nu * self.IsserlisPow12(Kxx)) - (self.rho * self.IsserlisPow12(K)) + (self.rho * self.IsserlisPow22(K))

    def CovCC(self, K, 
              Kt, Ktprime,
              Kxx, Kxprimexprime, Kttprime,
              Ktxprimexprime, Kxxtprime,
              Kxxxprimexprime,
            ):
        """ Covariance between collocation points """

        term1 = Kttprime - (self.nu * Ktxprimexprime) - (self.rho * Kt) + (self.rho * self.IsserlisPow12(Kt))
        term2 = (-self.nu * Kxxtprime) + (self.nu**2 * Kxxxprimexprime) + (self.nu * self.rho * Kxx) - (self.nu * self.rho * self.IsserlisPow12(Kxx))
        term3 = -(self.rho * self.CovUC(K, Ktprime, Kxprimexprime))
        term4 = (self.rho * self.CovU2C(K, Ktprime, Kxprimexprime))

        return term1 + term2 + term3 + term4


    def forward(self,
                x: Tensor,                                              # observation locations
                PerBoundaryLow: Tensor| None = None,                    # Lower boundary
                PerBoundaryUp: Tensor | None = None,                    # Upper boundary
                z: Tensor | None = None,                                # collocation locations
                adaptive_nugget: bool = False,                          # To compute an adaptive. nugget for stability: taken from Appendix A.1, https://arxiv.org/pdf/2103.12959.
                diag_blocks: bool = False,
        ) -> Tensor:
        """ returns the K matrix for RxnDiffusion """
        
        if self.kernel.diff_mode == False:
            raise Exception('Can\'t differentiate when kernel is not in `DiffMode` == True.')
        if x.ndim == 3 and z.ndim == 3:
            raise Exception('Please give unexpanded inputs `ndim` = 2 shaped `x` (observations) and `z` (collocation)')
        
        # Obs-Obs
        xsymm = expanding_inputs(x)
        
        if z == None and PerBoundaryLow == None and PerBoundaryUp == None:
            printRed('Provided with zero collocation points (simple Kriging K matrix)')
            return self.kernel(xsymm)
        
        # Obs-PerBoundaryLow
        x_BL, BL_x = expanding_inputs(x, PerBoundaryLow)
        
        # Obs-PerBoundaryLow
        x_BU, BU_x = expanding_inputs(x, PerBoundaryUp)
        
        # PerBoundaryLow-PerBoundaryLow
        BLsymm = expanding_inputs(PerBoundaryLow)
        
        # PerBoundaryLow-PerBoundaryUp
        BL_BU, BU_BL = expanding_inputs(PerBoundaryLow, PerBoundaryUp)
        
        # PerBoundaryLow-PerBoundaryLow
        BUsymm = expanding_inputs(PerBoundaryUp)
        
        # Obs-Colloc
        x_z, z_x = expanding_inputs(x, z)    
        
        # PerBoundaryLow-Colloc
        BL_z, z_BL = expanding_inputs(PerBoundaryLow, z)
        
        # PerBoundaryLow-Colloc
        BU_z, z_BU = expanding_inputs(PerBoundaryUp, z)
        
        # Colloc-Colloc
        zsymm = expanding_inputs(z)
        
        #--- First row ----
        K11 = self.kernel(xsymm)

        #--- Second row ----
        K21 = self.IsserlisPow12(K11)
        K22 = self.IsserlisPow22(K11)

        #--- Third row ----
        KBL = self.kernel(BL_x, x_BL)
        KBU = self.kernel(BU_x, x_BU)
        K31 = KBL - KBU
        K32 = self.IsserlisPow12(KBL) - self.IsserlisPow12(KBU)
            
        K33 = self.kernel(BLsymm) + self.kernel(BUsymm) - self.kernel(BL_BU, BU_BL) - self.kernel(BU_BL, BL_BU)

        # Unit test: Symmetry of covariance
        assert torch.allclose(self.kernel(BL_BU, BU_BL), self.kernel(BU_BL, BL_BU))

        #--- Fourth row ---- [BL-BU]^2 = 0 OR BL^2 - BU^2 = 0

        # BL^2 - BU^2 = 0
        if self.boundary == 'BL2 - BU2':
            K41 = self.IsserlisPow12(KBL) - self.IsserlisPow12(KBU)
            
            K42 = self.IsserlisPow22(KBL) - self.IsserlisPow22(KBU)
            
            K43 = (self.IsserlisPow12(self.kernel(BLsymm)) 
                    + self.IsserlisPow12(self.kernel(BUsymm)) 
                    - self.IsserlisPow12(self.kernel(BL_BU, BU_BL)) 
                    - self.IsserlisPow12(self.kernel(BU_BL, BL_BU))
                    )
            
            K44 = (self.IsserlisPow22(self.kernel(BLsymm)) 
                   + self.IsserlisPow22(self.kernel(BUsymm)) 
                   - self.IsserlisPow22(self.kernel(BL_BU, BU_BL)) 
                   - self.IsserlisPow22(self.kernel(BU_BL, BL_BU))
                )
        else:
            # (BL - BU)^2 = 0
            K41 = IsserlisPow12(KBL) 
            K42 = IsserlisPow22(KBL-KBU)
            K43 = IsserlisPow12(self.kernel(BLsymm))
            K44 = IsserlisPow22(K33)
        
        #--- Fifth row ----
        Kt = kernel_derivatives_2D(z_x, x_z, self.kernel, 
                                   [[1,0],
                                    [0,0]])
        Kxx2 = kernel_derivatives_2D(z_x, x_z, self.kernel,
                                    [[0,2],
                                    [0,0]])
        K = self.kernel(z_x, x_z)

        K51 = self.CovUC(K, Kt, Kxx2) 

        K52 = self.CovU2C(K, Kt, Kxx2)
        
        KtBL = kernel_derivatives_2D(z_BL, BL_z, self.kernel, 
                                    [[1,0],
                                    [0,0]])
        Kxx2BL = kernel_derivatives_2D(z_BL, BL_z, self.kernel, 
                                            [[0,2],
                                            [0,0]])
        KBL = self.kernel(z_BL, BL_z)

        KtBU = kernel_derivatives_2D(z_BU, BU_z, self.kernel, 
                                    [[1,0],
                                    [0,0]])
        Kxx2BU = kernel_derivatives_2D(z_BU, BU_z, self.kernel, 
                                            [[0,2],
                                            [0,0]])
        KBU = self.kernel(z_BU, BU_z)

        K53 = self.CovUC(KBL, KtBL, Kxx2BL) - self.CovUC(KBU, KtBU, Kxx2BU)

        if self.boundary == 'BL2 - BU2':
            # ===========
            # BL^2 - BU^2
            # ===========
            K54 = self.CovU2C(KBL, KtBL, Kxx2BL) - self.CovU2C(KBU, KtBU, Kxx2BU)

        else:
            #===========
            # (BL - BU)^2 = 0
            #===========
            K54 = (self.rho * IsserlisPow22(KBL - KBU))

        K = self.kernel(zsymm)
        Kt = kernel_derivatives_2D(zsymm, None, self.kernel, [[1,0], [0,0]], 0,)    # dK/dx_1  
        Ktprime = kernel_derivatives_2D(zsymm, None, self.kernel, [[0,0], [1,0]], 0,)    # dK/dz_1
        Ktt = kernel_derivatives_2D(zsymm, None, self.kernel, [[1,0], [1,0]], 0,)    # d2K/dx_1 dz_1 
        Ktxx = kernel_derivatives_2D(zsymm, None, self.kernel, [[1,0],[0,2]], 0)     # Diff. wrt. `z` first 
        Kxxt = kernel_derivatives_2D(zsymm, None, self.kernel, [[0,2],[1,0]], 0)     # Diff. wrt. `z` first
        Kx2x2 = kernel_derivatives_2D(zsymm, None, self.kernel, [[0,2], [0,2]], 0,)    # d4K/dx_2^2 dz_2^2
        Kxx = kernel_derivatives_2D(zsymm, None, self.kernel, [[0,2],[0,0]], 0)      # d2K/dx_2^2
        Kxprimexprime = kernel_derivatives_2D(zsymm, None, self.kernel, [[0,0],[0,2]], 0) # d2K/dz_2^2

        K55 = self.CovCC(K, 
                        Kt, Ktprime, 
                        Kxx, Kxprimexprime, Ktt, 
                        Ktxx, Kxxt,
                        Kx2x2
                    )

        # Sanity check
        symmetry_check(K55, matrix = 'K55')

        # whether to use u^2 observations or not
        if self.mode == 'u2':
            block_list = [K11, 
                        K21, K22,               # u^2
                        K31, K32, K33,
                        K41, K42, K43, K44,     # BL^2 - BU^2
                        K51, K52, K53, K54, K55]
            
            diag_list = [K11, K22, K33, K44, K55]
        
        else:
            block_list = [K11,  
                        K31, K33,
                        K51, K53, K55]
            
            diag_list = [K11, K33, K55]

        CoK = _symmetric_block(*block_list) 

        if self.mode == 'u2':
            row_1 = torch.cat([K11, K21.T, K31.T, K41.T, K51.T], dim = 1)
            row_2 = torch.cat([K21, K22, K32.T, K42.T, K52.T], dim = 1)
            row_3 = torch.cat([K31, K32, K33, K43.T, K53.T], dim = 1)
            row_4 = torch.cat([K41, K42, K43, K44, K54.T], dim = 1)
            row_5 = torch.cat([K51, K52, K53, K54, K55], dim = 1)

            assert torch.allclose(CoK, torch.cat([row_1, row_2, row_3, row_4, row_5]))

        # Sanity check
        symmetry_check(CoK, matrix='CoK')

        # Invoking adaptive nugget
        self._nugget = torch.eye(CoK.shape[0], dtype = CoK.dtype) 
        
        if adaptive_nugget:
            self._nugget = _adaptive_nugget_constructor(*diag_list)

        # Sanity check
        dim_check(CoK, self._nugget)

        # Defining a physicsmatrix object
        K = PhysicsMatrix(CoK, self._nugget)
        if diag_blocks:
            K = PhysicsMatrix(CoK, 
                            self._nugget, 
                            diag_list)

        return K

    def cross_covariance(self, x: Tensor, x_test: Tensor, 
                          PerBoundaryLow: Tensor | None = None,
                          PerBoundaryUp: Tensor | None = None,
                           z: Tensor | None = None,) -> Tensor:
        """ returns the H matrix for RxnDiffusion """
    
        if self.kernel.diff_mode == False:
            raise Exception('Can\'t differentiate when kernel is not in `DiffMode` == True.')
        if x.ndim == 3 and z.ndim == 3:
            raise Exception('Please give unexpanded inputs `ndim` = 2 shaped `x` (observations) and `z` (collocation)')

        # Obs-test
        x_e, x_test_e = expanding_inputs(x, x_test)
    
        if z == None and PerBoundaryLow == None and PerBoundaryUp == None:
            printRed('Provided with zero collocation points (simple Kriging H matrix)')
            return self.kernel(x_e, x_test_e)

        # BL-test
        BL_x_test, x_test_BL = expanding_inputs(PerBoundaryLow, x_test)

        # BU-test
        BU_x_test, x_test_BU = expanding_inputs(PerBoundaryUp, x_test)

        # Colloc-test
        z_test, test_z = expanding_inputs(z, x_test)
    
        #--- First row ---
        H1 = self.kernel(x_e, x_test_e)

        #--- Second row ---
        H2 = self.IsserlisPow12(H1)

        #--- Third row ---
        H3 = self.kernel(BL_x_test, x_test_BL) - self.kernel(BU_x_test, x_test_BU)

        #--- Fourth row ---
        if self.boundary == 'BL2 - BU2':
            #===========
            # BL^2 - BU^2
            #===========
            H4 = self.IsserlisPow12(self.kernel(BL_x_test, x_test_BL)) - self.IsserlisPow12(self.kernel(BU_x_test, x_test_BU))

        else:
            #===========
            # (BL - BU)^2
            #===========
            H4 = IsserlisPow12(self.kernel(BL_x_test, x_test_BL))

        #--- Fifth row ---
        Ht = kernel_derivatives_2D(z_test, test_z, self.kernel, [[1,0],[0,0]])
        Hxx = kernel_derivatives_2D(z_test, test_z, self.kernel, [[0,2],[0,0]])
        H = self.kernel(z_test, test_z)

        H5 = self.CovUC(H, Ht, Hxx)

        # print(f'{H1.size()=}', f'{H2.size()=}', f'{H3.size()=}')
        with torch.no_grad():
            # whether to use u^2 observations or not
            if self.mode == 'u2':
                block_list = [H1, H2, H3, H4, H5]
            else:
                block_list = [H1, H3, H5]
            
            CoH = torch.cat(block_list, dim = 1).double()

        return CoH

# Modying the centering class
# Centered observations
class RxnDiffCenteredValues(CenteredValues):
    """ Centering functions and Tensor that stores all of the centered information (the centering, the centered output, etc.) """
    def __init__(
        self,
        *args,
        rho: float = 1.0,
        stationary: bool = True,                                 # Stationary assumption for Y
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.rho = float(rho)
        self.stationary = bool(stationary)

    # kernel based centering
    def centering_k(self, x: Tensor, sigma: float = 1.0):
        """ returns k(x,x) """
        # Storing original
        original_val = self.kernel.diag_mode 
        self.kernel.diag_mode = True

        k = sigma**2 * self.kernel(x).squeeze()            

        self.kernel.diag_mode = original_val
        return k

    # centering with empirical mean 
    def centering_y(self, y: Tensor, mean: float | None = None):
        """ return `torch.mean(y)` """

        y_mean = 0.0
        if not self.stationary:
            if mean is None:
                y_mean = torch.mean(y)
            else:
                y_mean = mean
        return y_mean

    # Centering u^2 function
    def centering_y2(self, y: Tensor, x: Tensor, sigma = 1.0, mean = None):                       
        """ Centering the given observations Y^2. """
        
        # Centering formula -- EY^2 - (EY)^2 = k(x,x); EY^2 = k(x,x) + (EY)^2
        y_mean_sq = (self.centering_y(y, mean))**2

        k = self.centering_k(x, sigma)            

        return y_mean_sq, k

    # Centering collocation points
    def centering_z(self, y: Tensor, z: Tensor, sigma = 1.0, mean = None):                       
        """ Centering the collocation points: du/dt - nu * d2u/dx2 - rho * u + rho * u^2 """

        y_mean = self.centering_y(y, mean)

        term1 = (-self.rho * y_mean)
        y_mean_sq, k = self.centering_y2(y, z, sigma, mean)

        term2 = (self.rho * (y_mean_sq + k))
        return term1, term2


#*******************
# Main body()
#*******************
def main(rho = 5.0,                                                             # Reaction coefficient
         nu = 5.0,                                                              # Diffusion coefficient 
        nobs: int = 0,
        N_colloc: int = 900,
        Nx: int = 120, 
        Nt: int = 60,
        N_initial: int = 60,
        N_boundary: int = 120,
        kernel: Literal['RBF', 'Matern32', 'Matern52'] = 'RBF', 
        init_lengthscale: float | Tensor | None = None, 
        stationary: bool = True,                                                # stationary assumption
        jitter_co_Krig: float = 1e-6,                                           # jitter for numerical stability
        adaptive_nugget: bool = True,                                           # adaptive nugget option
        filtering: list = None,                                                 # LOOCV filter matrix
        subsampling: int = 5,                                                   # subsampling collocation points for faster training
        weight_decay: float = 1e-4,                                             # optimizer weight decay
        steps: int = 0,                                                         # optimizer steps
        learning_rate: float = 0.01,                                            # optimizer learning rate
        loss_landscape: bool = False,                                           # to visualize loss-landscape 
        best_jitter: bool = False,                                              # find best jitter 
        mode: str = 'u2',                                                       # u2 observations
        mean_explicit: float | None = None,                                     # Explicit mean value for the prior (far from observations)
        seed: int = 0,
        save: bool = False,                                                     # saving the model
        random_colloc: bool = False,                                            # To use random collocation points,
        sigma: float = 1.0,                                                     # Explicit sigma value to compute LOOCV
        **kwargs,
    ):

    u_true, train_x, PerBoundaryLower, PerBoundaryUpper, train_z, test_x, train_y, train_v = rxndiffusion_DoE(N_initial = N_initial, 
                                                                                                              N_boundary = N_boundary,
                                                                                                              N_colloc = N_colloc,
                                                                                                              random_colloc=random_colloc,
                                                                                                              Nx = Nx,
                                                                                                              Nt = Nt,
                                                                                                              nu = nu,
                                                                                                              rho = rho,
                                                                                                              N_rand = nobs,
                                                                                                              u0 = 'gauss',
                                                                                                              seed = seed)

    # choose kernel
    _kernel = choose_kernel(kernel)(lengthscale=init_lengthscale,
                                    DiffMode = True) if init_lengthscale is not None else choose_kernel(kernel)(DiffMode = True)

    # _kernel = AdditiveRBFKernel(lengthscale1 = torch.tensor([0.1, 0.4])) 
    
    # name while saving
    file_name = f'_init{_kernel.lengthscale.tolist()}_opt{steps}_rho{rho}_nu{nu}_colloc{N_colloc}_rand{random_colloc}_subsampling{subsampling}_lr{learning_rate}'

    # define co-Kriging model
    mean = (1 - stationary) * torch.mean(train_y) if mean_explicit is None else (1 - stationary) * mean_explicit
    CoKrig = RxnDiffusionRegressor(kernel = _kernel,
                                   jitter = jitter_co_Krig,
                                   nu = nu,
                                   rho = rho,
                                   mean = mean,
                                   true_fn = u_true,
                                   mode = 'u2',
                                   save = save,
                                )

    print(f'u_mean used for covariance computations and centering: {CoKrig.mean} \n')

    # Centered observations
    y_train = RxnDiffCenteredValues(kernel=_kernel,
                                    rho = rho,
                                    stationary=stationary
                                )
    
    y_mean = y_train.centering_y(train_y, mean=mean_explicit)
    y_train.forward(train_y, y_mean)
    
    y2_train = RxnDiffCenteredValues(kernel=_kernel,
                                    rho = rho,
                                    stationary=stationary
                                )
    y_mean_sq, k = y2_train.centering_y2(train_y, train_x, sigma=sigma, mean=mean_explicit)
    y2_train.forward(train_y**2, k + y_mean_sq)
    
    z_train = RxnDiffCenteredValues(kernel=_kernel,
                                    rho = rho,
                                    stationary=stationary
                                )
    term1, term2 = z_train.centering_z(train_y, train_z, sigma=sigma, mean=mean_explicit)
    z_train.forward(train_v, term1 + term2)

    ### Concated obs
    list_obs = [y_train.centered, y2_train.centered, torch.zeros(PerBoundaryLower.size(dim=0)), 
                torch.zeros(PerBoundaryLower.size(dim=0)), z_train.centered]
    red_list_obs = [y_train.centered, y2_train.centered, 
                    torch.zeros(PerBoundaryLower.size(dim=0)), 
                    torch.zeros(PerBoundaryLower.size(dim=0)), z_train.centered[::subsampling]]
    if mode == 'u':
        list_obs = [y_train.centered, torch.zeros(PerBoundaryLower.size(dim=0)), z_train.centered]
        red_list_obs = [y_train.centered, torch.zeros(PerBoundaryLower.size(dim=0)), z_train.centered[::subsampling]]

    ConcatedObs = torch.cat(list_obs).reshape(-1).double()
    ConcatedObsReduced = torch.cat(red_list_obs).reshape(-1).double()

    # Training the co-Kriging model
    training_model = fit_regressor(CoKrig,
                                   steps = steps,
                                   learning_rate = learning_rate,
                                )

    loss_fn_args, loss_fn_kwargs = (train_x, 
                                    PerBoundaryLower, 
                                    PerBoundaryUpper, 
                                    train_z[::subsampling],), dict(jitter = 1e-4, 
                                                                ConcatedObs = ConcatedObsReduced, 
                                                                filtering = filtering,
                                                                adaptive_nugget = adaptive_nugget)

    training_model.central_differences_train(*loss_fn_args,
                                            weight_decay = weight_decay,
                                            **loss_fn_kwargs
                                        )
    
    training_model.plot_loss()

    # Computing the sigma
    print(f'LOOCV optimal sigma: {CoKrig.LOOCVloss(*loss_fn_args, mode = "sigma", **loss_fn_kwargs,).item()}')
    
    # Recomputing ConcatedObs with LOOCV sigma
    y_mean_sq, k = y2_train.centering_y2(train_y, train_x, sigma=CoKrig.LOOCVsigma, mean=mean_explicit)
    y2_train.forward(train_y**2, k + y_mean_sq)
    
    term1, term2 = z_train.centering_z(train_y, train_z, sigma=CoKrig.LOOCVsigma, mean=mean_explicit)
    z_train.forward(train_v, term1 + term2)

    list_obs = [y_train.centered, y2_train.centered, torch.zeros(PerBoundaryLower.size(dim=0)), 
                torch.zeros(PerBoundaryLower.size(dim=0)), z_train.centered]
    red_list_obs = [y_train.centered, y2_train.centered, 
                    torch.zeros(PerBoundaryLower.size(dim=0)), 
                    torch.zeros(PerBoundaryLower.size(dim=0)), z_train.centered[::subsampling]]
    if mode == 'u':
        list_obs = [y_train.centered, torch.zeros(PerBoundaryLower.size(dim=0)), z_train.centered]
        red_list_obs = [y_train.centered, torch.zeros(PerBoundaryLower.size(dim=0)), z_train.centered[::subsampling]]

    ConcatedObs = torch.cat(list_obs).reshape(-1).double()
    ConcatedObsReduced = torch.cat(red_list_obs).reshape(-1).double()

    # Loss landscape visualization
    if loss_landscape:
        CoKrig.loss_landscape_2D(*loss_fn_args, 
                                **loss_fn_kwargs,
                                txmin = 0.1,
                                txmax = 0.5,
                                tymin = 0.1,
                                # tymax = 2*math.pi,
                                tymax = 0.5,
                                Nx = 20, Ny = 20,
                                name = file_name)

    # Trained predictions
    predict_kwargs = dict(x_train = train_x, PerBoundaryLow = PerBoundaryLower, 
                          PerBoundaryUp = PerBoundaryUpper, 
                        x_test = test_x, z_train = train_z,
                        ConcatedObs = ConcatedObs, centering = y_train.centering,
                        cholesky_mode = True, adaptive_nugget = adaptive_nugget,
                        sigma = CoKrig.LOOCVsigma, best_jitter = best_jitter,
                )
    prediction, std_dev = CoKrig.predict(**predict_kwargs
                                    )

    # training simple Kriging on observations of y
    lengthscale_dict = dict(lengthscale = init_lengthscale) if init_lengthscale is not None else None
    args, kwargs = (train_x, train_y, test_x,), dict(kernel_cls = choose_gpy_kernel(kernel),       # Simple Kriging kernel
                                                    nu = choose_gpy_kernel.nu,
                                                    ard_num_dims = 2, 
                                                    best_jitter = best_jitter) 
    
    if lengthscale_dict is not None: 
        training_model.gpy_train(*args, 
                                **kwargs, 
                                **lengthscale_dict) 
    else:
        training_model.gpy_train(*args, **kwargs)

    # # plotting the loss curve
    # training_model.plot_loss_SK()
    
    # passing the predictions directly
    training_model.pass_to_model(training_model.predictionSK,
                                 training_model.lowerSK,
                                training_model.upperSK
                            )

    # computing the metrics for the squared prediction
    training_model.metrics_SK(training_model.predictionSK, u_true)

    # Visualizing the simple Kriging solution
    # CoKrig.visualize_2D(train_x, test_x, train_y,
    #                     option = 'SK', levels = 100, cmap = 'rainbow')

    CoKrig.visualize_2D(train_x, 
                        test_x, 
                        train_y, 
                        cmap = 'rainbow',
                        option = 'true',
                        levels = 100,
                        name = file_name,
                        # limit = True
                    )
    
    CoKrig.visualize_2D(train_x,
                        test_x,
                        train_y,
                        option = 'CK',
                        levels = 100, cmap = 'rainbow',
                        name = file_name,
                        # limit = True
                    )

    CoKrig.visualize_2D(train_x, 
                        test_x, 
                        train_y, 
                        cmap = 'rainbow',
                        levels = 100,
                        option = 'UQ',
                        name = file_name,
                        # limit = True
                    )

    CoKrig.visualize_2D(train_x, 
                        test_x, 
                        train_y, 
                        cmap = 'rainbow',
                        levels = 100,
                        name = file_name,
                        # limit = True
                    )
    
    

    # saving the model [optional]
    if save and best_jitter:
        CoKrig.save_model(directory_name=f'{CoKrig.__class__.__name__}Models',
                          jitter = CoKrig._bestjitter,
                        problem_name = f'{CoKrig.kernel.__class__.__name__}_bestjitt_{file_name}'
                )
    if save and not best_jitter:
        CoKrig.save_model(directory_name=f'{CoKrig.__class__.__name__}Models',
                          jitter = CoKrig.jitter,
                        problem_name = f'{CoKrig.kernel.__class__.__name__}_{file_name}'
                )

    print(100*'='+'\n'+'Co-Kriging metrics'+'\n'+100*'=')
    print(f'RMSE: {CoKrig.RMSE.item()}')
    print(f'MAE: {CoKrig.MAE.item()}')
    print(f'L2relative: {CoKrig.L2RELATIVE.item()}\n')
    print(100*'='+'\n'+'Simple Kriging metrics'+'\n'+100*'=')
    print(f'RMSE: {CoKrig.RMSE_SK.item()}')
    print(f'MAE: {CoKrig.MAE_SK.item()}')
    print(f'L2relative: {CoKrig.L2RELATIVE_SK.item()}')

if __name__ == '__main__':
    main(
        rho = 5.0,
        nu = 5.0,
        jitter_co_Krig = 1e-4,
        init_lengthscale=torch.tensor([0.5, 0.5]),
        steps = 500,
        adaptive_nugget = True,
        subsampling = 2,
        nobs = 0,
        N_initial = 60, 
        N_boundary = 60,
        Nx = 60,
        Nt = 60,
        filtering = [1,0,0,0,0], 
        # learning_rate = 0.1,
        weight_decay = 0.0,
        stationary = False,
        N_colloc = 900,
        kernel = 'RBF',
        best_jitter = True,
        # save=True,
        random_colloc=True,
        # loss_landscape=True,
    )



    

    
