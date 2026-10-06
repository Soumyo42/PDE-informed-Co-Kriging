# imports 
import sys
import torch

sys.path.append('..')           # To search in the right directory 

from physics import PhysicsRegression, PhysicsMatrix, _adaptive_nugget_constructor, _symmetric_block, CenteredValues, fit_regressor
from kernels import kernel_derivatives_2D, expanding_inputs, printRed, symmetry_check, dim_check, choose_kernel
from SimpleKrigGpytorch import choose_gpy_kernel
from torch import Tensor
import numpy as np
import math
from typing import Literal
import matplotlib.pyplot as plt

# Analytical solution function via: https://github.com/a1k12/characterizing-pinns-failure-modes/blob/main/pbc_examples/systems_pbc.py

#==================
# NOT TESTED
#==================
# # CUDA support
# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')
#==================

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

def convection_diffusion(u0: str, nu, beta, source=0, xgrid=30, nt=30):         # OG xgrid = 256, nt = 100
    """Calculate the u solution for convection/diffusion, assuming PBCs.
    Args:
        u0: Initial condition
        nu: viscosity coefficient
        beta: wavespeed coefficient
        source: q (forcing term), option to have this be a constant
        xgrid: size of the x grid
    Returns:
        u_vals: solution
    """

    N = xgrid
    h = 2 * np.pi / N
    x = np.arange(0, 2*np.pi, h) # not inclusive of the last point
    t = np.linspace(0, 1, nt).reshape(-1, 1)
    X, T = np.meshgrid(x, t)

    # call u0 this way so array is (n, ), so each row of u should also be (n, )
    u0 = function(u0)
    u0 = u0(x)

    G = (np.copy(u0)*0)+source # G is the same size as u0

    IKX_pos = 1j * np.arange(0, N/2+1, 1)
    IKX_neg = 1j * np.arange(-N/2+1, 0, 1)
    IKX = np.concatenate((IKX_pos, IKX_neg))
    IKX2 = IKX * IKX

    uhat0 = np.fft.fft(u0)
    nu_factor = np.exp(nu * IKX2 * T - beta * IKX * T)
    A = uhat0 - np.fft.fft(G)*0 # at t=0, second term goes away
    uhat = A*nu_factor + np.fft.fft(G)*T # for constant, fft(p) dt = fft(p)*T
    u = np.real(np.fft.ifft(uhat))

    u_vals = u.flatten()
    return u_vals

#==================================
# Example use-case
#==================================
# # Saving the solution in an array
# usoln = convection_diffusion('sin(x)', 0, 30).reshape(60, 60)
#==================================

# modifying the forward function and the cross-covariance functions
class ConvectionRegression(PhysicsRegression):
    r"""
    Convection equation co-Kriging class. 
    
    Define the following methods:
        1. `.forward()`: this is going to give the observation covariance matrix.
        2. `.cross_covariance(mode:str='u')`: define one or multiple modes. 
    """
    def __init__(
        self,
        *args,
        beta: float = 30,
        mean: float = 0.0,
        diagnostic: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, beta = beta, **kwargs)

        self.mean = float(mean)
        self.diagnostic = bool(diagnostic)

    def forward(self, x, 
                bL: Tensor | None = None,                               # boundary conditions (lower)
                bU: Tensor | None = None,                               # boundary conditions (upper)
                z: Tensor | None = None,                                # Dummy variable 
                adaptive_nugget = False, 
                diag_blocks = True,
                )-> PhysicsMatrix:
        """ returns the K matrix """

        if self.kernel.diff_mode == False:
            raise Exception('Can\'t differentiate when kernel is not in `DiffMode` == True.')
        if x.ndim == 3 and z.ndim == 3:
            raise Exception('Please give unexpanded inputs `ndim` = 2 shaped `x` (observations) and `z` (collocation)')

        # Obs-Obs
        xsymm = expanding_inputs(x)

        if z == None and bL == None and bU == None:
            printRed('Provided with zero collocation points (simple Kriging K matrix)')
            return self.kernel(xsymm)

        # Obs-PerBoundaryLow
        x_BL, BL_x = expanding_inputs(x, bL)
        
        # Obs-PerBoundaryUp
        x_BU, BU_x = expanding_inputs(x, bU)
        
        # PerBoundaryLow-PerBoundaryLow
        BLsymm = expanding_inputs(bL)
        
        # PerBoundaryLow-PerBoundaryUp
        BL_BU, BU_BL = expanding_inputs(bL, bU)
        
        # PerBoundaryLow-PerBoundaryLow
        BUsymm = expanding_inputs(bU)
        
        # Obs-Colloc
        x_z, z_x = expanding_inputs(x, z)    
        
        # PerBoundaryLow-Colloc
        BL_z, z_BL = expanding_inputs(bL, z)
        
        # PerBoundaryLow-Colloc
        BU_z, z_BU = expanding_inputs(bU, z)
        
        # Colloc-Colloc
        zsymm = expanding_inputs(z)

        # --- First row ---
        K11 = self.kernel(xsymm)
        
        # Sanity check
        symmetry_check(K11, matrix = 'K11')

        #--- Second row ----
        K21 = self.kernel(BL_x, x_BL) - self.kernel(BU_x, x_BU)
        K22 = self.kernel(BLsymm) + self.kernel(BUsymm) - self.kernel(BL_BU, BU_BL) - self.kernel(BU_BL, BL_BU)

        # Unit test: Symmetry of covariance
        assert torch.allclose(self.kernel(BL_BU, BU_BL), self.kernel(BU_BL, BL_BU))

        #--- Third row ----
        Kt = kernel_derivatives_2D(z_x, x_z, self.kernel, 
                                   [[1,0],
                                    [0,0]])
        Kx = kernel_derivatives_2D(z_x, x_z, self.kernel,
                                    [[0,1],
                                    [0,0]])
        K31 = Kt + self.beta * Kx
        
        KBLt = kernel_derivatives_2D(z_BL, BL_z, self.kernel,
                              index = [[1,0],
                                       [0,0]])
        KBUt = kernel_derivatives_2D(z_BU, BU_z, self.kernel,
                              index = [[1,0],
                                       [0,0]])

        KBLx = kernel_derivatives_2D(z_BL, BL_z, self.kernel,
                              index = [[0,1],
                                       [0,0]])
        KBUx = kernel_derivatives_2D(z_BU, BU_z, self.kernel,
                              index = [[0,1],
                                       [0,0]])
        K32 = KBLt - KBUt + self.beta * (KBLx - KBUx)

        Ktt = kernel_derivatives_2D(zsymm, None, self.kernel, [[1,0], [1,0]],)    # d2K/dx_1 dz_1   
        Kxx = kernel_derivatives_2D(zsymm, None, self.kernel, [[0,1], [0,1]],)    # d2K/dx_2 dz_2
        Ktx = kernel_derivatives_2D(zsymm, None, self.kernel, [[1,0],[0,1]],)     
        Kxt = kernel_derivatives_2D(zsymm, None, self.kernel, [[0,1],[1,0]],)

        K33 = Ktt + (self.beta * Ktx) + (self.beta * Kxt) + (self.beta**2 * Kxx)

        # Sanity check
        symmetry_check(K33, matrix = 'K33')

        block_list = [K11, 
                    K21, K22, 
                    K31, K32, K33]
        
        diag_list = [K11, K22, K33]

        CoK = _symmetric_block(*block_list) 

        # Sanity check
        assert torch.allclose(CoK, torch.cat([torch.cat([K11, K21.T, K31.T], dim = 1),
                                            torch.cat([K21, K22, K32.T], dim = 1),
                                            torch.cat([K31, K32, K33], dim = 1)]))

        # Invoking adaptive nugget
        self._nugget = torch.eye(CoK.shape[0], dtype = CoK.dtype) 

        if adaptive_nugget:
            self._nugget = _adaptive_nugget_constructor(*diag_list)

        # Sanity check
        symmetry_check(CoK, matrix = 'CoK')

        # Sanity check
        dim_check(CoK, self._nugget)

        # Defining a physicsmatrix object
        K = PhysicsMatrix(CoK, self._nugget)
        if diag_blocks:
            K = PhysicsMatrix(CoK, 
                            self._nugget, 
                            diag_list)
        return K

    # cross covariance for predicting u
    def cross_covariance(self, x, 
                         x_test,
                         bL = None,
                         bU = None,
                         z = None):
        """ returns the H matrix for predicting u """

        # --- first row ---
        x_t, t_x = expanding_inputs(x, x_test)
        H1 = self.kernel(x_t, t_x)

        if z == None:
            printRed('provided with zero collocation points (simple Kriging H matrix)')
            return H1

        # --- second row --- 
        # BL-test
        BL_x_test, x_test_BL = expanding_inputs(bL, x_test)
        
        # BU-test
        BU_x_test, x_test_BU = expanding_inputs(bU, x_test)

        H2 = self.kernel(BL_x_test, x_test_BL) - self.kernel(BU_x_test, x_test_BU)

        # --- third row ---
        z_t, t_z = expanding_inputs(z, x_test)
        Ht = kernel_derivatives_2D(z_t, t_z, self.kernel,
                                   index = [[1,0],
                                            [0,0]])
        Hx = kernel_derivatives_2D(z_t, t_z, self.kernel,
                                   index = [[0,1],
                                            [0,0]])
        H3 = Ht + self.beta * Hx

        block_list = [H1, H2, H3]

        CoH = torch.cat(block_list, dim = 1).double()

        return CoH

# Modying the centering class
# Centered observations
class ConvectionCenteredValues(CenteredValues):
    """ Centering functions and Tensor that stores all of the centered information (the centering, the centered output, etc.) """
    def __init__(
        self,
        *args,
        stationary: bool = True,                                 # Stationary assumption for Y
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.stationary = bool(stationary)

    # centering with empirical mean 
    def centering_y(self, y: Tensor):
        """ return `torch.mean(y)` """

        y_mean = 0.0
        if not self.stationary:
            y_mean = torch.mean(y)

        return y_mean

# The design of experiments
def Convection_DoE(beta: float = 30, N_initial: int = 60,
                     Nx: int  = 30, Nt: int  = 30, N_boundary: int = 60,
                     N_rand: int = 0, N_colloc: int = 30,
                     random_colloc: bool = False, 
                     seed: int = 0, t_obs: float | None = None,
                     u0: str = 'sin(x)', 
                     **kwargs):

    generator = torch.Generator().manual_seed(seed)

    # Calling the solution function
    u_true = convection_diffusion(u0 = u0, nu = 0, beta = beta,
                                xgrid = Nx, nt = Nt).reshape(Nt, Nx)
    
    X_min = 0.0; X_max = 2*np.pi
    T_min = 0.0; T_max = 1.0

    dx = (X_max - X_min)/Nx

    # # Visualize
    # x = np.arange(X_min, X_max, dx) # not inclusive of the last point
    # t = np.linspace(T_min, T_max, Nt).reshape(-1, 1)
    # X, T = np.meshgrid(x, t)
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
    dx = (X_max - X_min) / N_initial
    X_range = torch.arange(X_min, X_max, dx)
    Initial = torch.stack([torch.zeros(X_range.size(dim=0)), 
                           X_range],
                           dim = 1)
    u0 = function(u0)
    u_initial = u0(X_range.numpy())
    y_Initial = torch.tensor(u_initial).double()

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
                    torch.meshgrid(T_range, X_range, indexing='ij'),
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

    # # Visualize
    # plt.figure(figsize = (20,10))
    # plt.contourf(x_test[:,0].reshape(*u_true.size()).cpu(), 
    #              x_test[:,1].reshape(*u_true.size()).cpu(), 
    #              u_true.cpu(), cmap = 'rainbow', levels = 100)
    # plt.colorbar()
    # plt.title('True function')
    # plt.show()
    
    return u_true, x_train.double(), BoundaryLower.double(), BoundaryUpper.double(), Colloc.double(), x_test.double(), y_train.double(), y_colloc.double()

#*************************
# main body
#*************************

def main(beta = 30.0,                                                           # Convection coefficient 
        nobs: int = 0,
        N_colloc: int = 900,
        kernel: Literal['RBF', 'Matern32', 'Matern52'] = 'RBF', 
        init_lengthscale: float | Tensor | None = None, 
        stationary: bool = True,                                                # stationary assumption
        jitter_co_Krig: float = 1e-6,                                           # jitter for numerical stability
        adaptive_nugget: bool = False,                                          # adaptive nugget option
        filtering: list = [1,0,0],                                              # LOOCV filter matrix
        subsampling: int = 5,                                                   # subsampling collocation points for faster training
        weight_decay: float = 1e-4,                                             # optimizer weight decay
        steps: int = 0,                                                         # optimizer steps
        learning_rate: float = 0.01,                                            # optimizer learning rate 
        Nx: int = 30, 
        Nt: int = 30,
        best_jitter: bool = False,                                              # find best jitter 
        loss_landscape: bool = False,                                           # to visualize loss-landscape
        save: bool = False,                                                     # to save the model and figures
        seed: int = 0,
        random_colloc: bool = False,                                            # uniform random collocation points or not
        **kwargs,   
    ):

    # DoE
    u_true, train_x, bL, bU, train_z, test_x, train_y, train_v = Convection_DoE(beta=beta, 
                                                                        N_rand = nobs, 
                                                                        N_colloc = N_colloc,
                                                                        Nx = Nx,
                                                                        Nt = Nt,
                                                                        seed = seed,
                                                                        random_colloc=random_colloc
                                                                    )               

    # choose kernel
    _kernel = choose_kernel(kernel)(lengthscale=init_lengthscale,
                                    DiffMode = True) if init_lengthscale is not None else choose_kernel(kernel)(DiffMode = True) 

    # name while saving
    file_name = f'_init{_kernel.lengthscale.tolist()}_opt{steps}_beta{beta}_colloc{N_colloc}_rand{random_colloc}_subsampling{subsampling}_lr{learning_rate}'
    
    # define co-Kriging model
    mean = (1 - stationary) * torch.mean(train_y)
    CoKrig = ConvectionRegression(kernel = _kernel,                               # initialization co-Kriging
                                beta = beta,
                                mean = mean,
                                jitter = jitter_co_Krig,
                                true_fn = u_true,
                                save = save,
                            )

    print(f'u_mean used for covariance computations and centering: {CoKrig.mean}')

    # Centered observations
    y_train = ConvectionCenteredValues(stationary=stationary
                                )
    y_mean = y_train.centering_y(train_y)
    y_train.forward(train_y, y_mean)

    ### Concated obs
    ConcatedObs = torch.cat([y_train.centered,                          # random observations + initial condition
                            torch.zeros(bL.size(dim=0)),                # boundary condition
                            train_v]).reshape(-1).double()              # collocation points

    # Training the co-Kriging model
    training_model = fit_regressor(CoKrig,
                                   steps = steps,
                                   learning_rate=learning_rate,
                                )

    ### Concated obs reduced for training
    ConcatedObsReduced = torch.cat([y_train.centered,
                                    torch.zeros(bL.size(dim=0)),                # boundary condition
                                    train_v[::subsampling]]).reshape(-1).double()

    loss_fn_args, loss_fn_kwargs = (train_x, 
                                    bL, 
                                    bU, 
                                    train_z[::subsampling],), dict(jitter = 1e-6, 
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

    # Loss landscape visualization
    if loss_landscape:
        CoKrig.loss_landscape_2D(*loss_fn_args, 
                                **loss_fn_kwargs,
                                txmin = 0.1,
                                txmax = 1.0,
                                tymin = 0.1,
                                tymax = 2*math.pi,
                                Nx = 10, Ny = 20,
                                name = file_name,
                            )
    
    # Trained predictions
    predict_kwargs = dict(x_train = train_x, bL = bL, bU = bU, 
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

    # # plotting loss curve
    # training_model.plot_loss_SK()
    
    # passing the predictions directly
    training_model.pass_to_model(training_model.predictionSK,
                                 training_model.lowerSK,
                                training_model.upperSK
                            )
    
    # computing the metrics for the squared prediction
    training_model.metrics_SK(training_model.predictionSK, u_true)

    # Visualizations

    CoKrig.visualize_2D(train_x, 
                        test_x, 
                        train_y, 
                        option = 'true', cmap = 'rainbow',
                        levels = 100,
                        name = file_name,
                        )

    #===========================================
    # Simple Kriging predictions to compare with
    #===========================================
    # CoKrig.visualize_2D(train_x, test_x, train_y,
    #                     option = 'SK', levels = 100, cmap = 'rainbow',
    #                     name = file_name,
    #                 )
    #===========================================

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
                        option = 'UQ',
                        levels = 100,
                        name = file_name,
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
        beta = 30,
        jitter_co_Krig=1e-9,
        Nx = 60,
        Nt = 60,
        init_lengthscale=torch.tensor([0.5, 0.5]),
        steps = 500,
        adaptive_nugget = True,
        subsampling = 2,
        # filtering = [1,0,0], 
        # learning_rate = 0.1,
        weight_decay = 0.0,
        best_jitter = True,
        # stationary = False,
        N_colloc = 900,
        kernel = 'RBF',
        # loss_landscape=True,
        save=True,
        random_colloc=True
    )