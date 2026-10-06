# imports 
import sys
import torch

sys.path.append('..')           # To search in the right directory 

from physics import PhysicsRegression, PhysicsMatrix, _adaptive_nugget_constructor, _symmetric_block, CenteredValues, fit_regressor
from kernels import kernel_derivatives_1D, expanding_inputs, printRed, IsserlisPow2, symmetry_check, dim_check
from kernels import IsserlisPow12, IsserlisPow22, ProductKernel, choose_kernel
from SimpleKrigGpytorch import choose_gpy_kernel
from torch import Tensor
import math
from typing import Literal

# modifying the forward function and the cross-covariance functions
class LogisticRegression(PhysicsRegression):
    r"""
    Logistic equation co-Kriging class. 
    
    Define the following methods:
        1. `.forward()`: this is going to give the observation covariance matrix.
        2. `.cross_covariance(mode:str='u')`: define one or multiple modes. 
    """
    def __init__(
        self,
        *args,
        rho: float = 1.0, 
        mean: float = 0.0,
        diagnostic: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, rho = rho, **kwargs)

        self.mean = float(mean)
        self.diagnostic = bool(diagnostic)

    def forward(self, x, 
                z: Tensor | None = None,                                # Dummy variable 
                adaptive_nugget = False, 
                diag_blocks = True,
                )-> PhysicsMatrix:
        """ returns the K matrix """

        # --- first row ---
        xsymm = expanding_inputs(x)

        K11 = self.kernel(xsymm)

        # Sanity check
        symmetry_check(K11, matrix = 'K11')

        #================================================================
        # DIAGNOSTICS
        #================================================================
        if self.diagnostic and not isinstance(self.kernel, ProductKernel):
            raise ValueError('Cannot perform diagnostics without ProductKernel')

        if self.diagnostic and not self.kernel.lengthscale == math.pi * torch.ones(x.size(dim=-1)):
            raise ValueError('Diagnostics designed for fixed lengthscale [math.pi * torch.ones(x.size(dim=0))]')

        if self.diagnostic:
            x_diagnostic = x / (math.pi * torch.ones(x.size(dim=-1)))
            z_diagnostic = z / (math.pi * torch.ones(x.size(dim=-1)))

            xsymm_diagnostic = expanding_inputs(x_diagnostic)

            assert torch.allclose(K11, 
                                  ((1 + xsymm_diagnostic**2) 
                                   * (1 + xsymm_diagnostic.transpose(0,1)**2)).prod(dim=-1)
                                )
        #================================================================

        #================================================================

        if z == None:
            printRed('provided with zero collocation points (simple Kriging K matrix)')
            self._nugget = torch.eye(K11.shape[0], dtype = K11.dtype) 
            return PhysicsMatrix(K11, self._nugget)

        # --- Second row --- 
        K21 = IsserlisPow12(K11, self.mean)
        K22 = IsserlisPow22(K11, self.mean)

        # Sanity check
        symmetry_check(K21, matrix = 'K21')

        # Sanity check
        symmetry_check(K22, matrix = 'K22')

        #================================================================
        # DIAGNOSTICS
        #================================================================
        if self.diagnostic:
            K11_diagnostic = ((1 + xsymm_diagnostic**2) 
                                * (1 + xsymm_diagnostic.transpose(0,1)**2)).prod(dim=-1)
            assert torch.allclose(K21,
                            2 * self.mean * K11_diagnostic
                        )

            assert torch.allclose(K22,
                           (2 * K11_diagnostic**2 + 4 * self.mean**2 * K11_diagnostic)
                        )
        #================================================================
        
        #================================================================
        
        # --- Third row ---
        x_z, z_x = expanding_inputs(x, z)
        Kt = kernel_derivatives_1D(z_x, x_z, self.kernel,
                                   index = [1,0])

        #================================================================
        # DIAGNOSTICS
        #================================================================
        if self.diagnostic:
            x_z_diagnostic, z_x_diagnostic = expanding_inputs(x_diagnostic, z_diagnostic)

            assert torch.allclose(Kt,
                        (2 * (1 / math.pi) * z_x_diagnostic * (1 + x_z_diagnostic**2)).prod(dim=-1)
                    )
        #================================================================

        #================================================================

        K = self.kernel(z_x, x_z)
        #================================================================
        # DIAGNOSTICS
        #================================================================
        if self.diagnostic:
            assert torch.allclose(K,
                        ((1 + z_x_diagnostic**2) * (1 + x_z_diagnostic**2)).prod(dim=-1)
                    )
        #================================================================
        
        #================================================================
        K31 = Kt - (self.rho * K) + (self.rho * IsserlisPow12(K, self.mean))
        #================================================================
        # DIAGNOSTICS
        #================================================================
        if self.diagnostic:
            Kt_diagnostic = (2 * (1 / math.pi) * z_x_diagnostic * (1 + x_z_diagnostic**2)).prod(dim=-1)
            K_diagnostic = ((1 + z_x_diagnostic**2) * (1 + x_z_diagnostic**2)).prod(dim=-1)

            assert torch.allclose(Kt, 
                                Kt_diagnostic)

            assert torch.allclose(K, K_diagnostic)

            # try: assert torch.allclose(Kt - (self.rho * K),
            #                       Kt_diagnostic - (self.rho * K_diagnostic), 
            #                       atol = 1e-6)
            # except: 
            #     print(f'max error: {torch.max(                                                  # Sometimes max error can go upto 1e-7. Maybe it's a float() vs. double() issue ?
            #                         torch.abs(Kt - (self.rho * K) 
            #                                   - (Kt_diagnostic - (self.rho * K_diagnostic)))
            #                     )
            #                 }'
            #         )
            #     raise AssertionError

            try: 
                assert torch.allclose(K31,
                        Kt_diagnostic - (self.rho * K_diagnostic) + (2 * self.mean * self.rho * K_diagnostic), 
                        atol = 1e-6
                    )
            except AssertionError:
                print(f'K31 = {K31}')
                # plt.imshow(K31.detach().numpy())
                # plt.colorbar()
                # plt.show()
                print(f'diagnostic = {Kt_diagnostic - (self.rho * K_diagnostic) + (2 * self.mean * self.rho * K_diagnostic)}')
                # plt.imshow((Kt_diagnostic - (self.rho * K_diagnostic) + (2 * self.mean * self.rho * K_diagnostic)).cpu())
                # plt.colorbar()
                # plt.show()
                raise AssertionError('Mismatch')
        #================================================================
        
        #================================================================
        
        K32 = IsserlisPow12(Kt, self.mean) - (self.rho * IsserlisPow12(K, self.mean)) + (self.rho * IsserlisPow22(K, self.mean))

        #================================================================
        # DIAGNOSTICS
        #================================================================
        if self.diagnostic:
            assert torch.allclose(K32,
                    (
                        (2 * self.mean * Kt_diagnostic) 
                        - (2 * self.rho * self.mean * K_diagnostic) 
                        + (self.rho * (2 * K_diagnostic**2  + 4 * self.mean**2 * K_diagnostic))
                    )
                )
        #================================================================
        
        #================================================================

        zsymm = expanding_inputs(z)
        K = self.kernel(zsymm)

        symmetry_check(K, matrix = 'K -- zsymm')

        Ktt = kernel_derivatives_1D(zsymm, None, self.kernel, 
                                    index = [1,1], arg = 0)

        #================================================================
        # DIAGNOSTICS
        #================================================================
        if self.diagnostic:
            zsymm_diagnostic = expanding_inputs(z_diagnostic)
            try:
                symmetry_check(4 * (1 / math.pi**2) * (zsymm_diagnostic * zsymm_diagnostic.transpose(0, 1)).prod(dim=-1))
                assert torch.allclose(Ktt,
                    4 * (1 / math.pi**2) * (zsymm_diagnostic * zsymm_diagnostic.transpose(0, 1)).prod(dim=-1) 
                )
            except AssertionError:
                print(f'Ktt = {Ktt}')
                print(f'diagnostic = {4 * (1 / math.pi**2) * (zsymm_diagnostic * zsymm_diagnostic.transpose(0, 1)).prod(dim=-1)}')
                raise AssertionError('Mismatch')
        #================================================================
        
        #================================================================

        # Sanity check
        assert torch.allclose(Ktt,  kernel_derivatives_1D(zsymm, None, self.kernel, 
                                            index = [1,1], arg = 1)
                        )

        Kt = kernel_derivatives_1D(zsymm, None, self.kernel, 
                                    index = [1,0])
        #================================================================
        # DIAGNOSTICS
        #================================================================
        if self.diagnostic:
            assert torch.allclose(Kt,
                    (2 * (1 / math.pi) * zsymm_diagnostic * (1 + zsymm_diagnostic.transpose(0, 1)**2)).prod(dim=-1)
                )
        #================================================================
        
        #================================================================
        Ktdash = kernel_derivatives_1D(zsymm, None, self.kernel, 
                                    index = [0,1])
        #================================================================
        # DIAGNOSTICS
        #================================================================
        if self.diagnostic:
            assert torch.allclose(Ktdash,
                    (2 * (1 / math.pi) * zsymm_diagnostic.transpose(0, 1) * (1 + zsymm_diagnostic**2)).prod(dim=-1)
                )
        #================================================================
        
        #================================================================

        # Sanity check
        symmetry_check(Ktt, matrix = 'Ktt -- K33')

        # Sanity check
        symmetry_check(K, matrix = 'K -- K33')

        # Sanity check
        symmetry_check(Kt + Ktdash, matrix = 'Kt + Ktdash -- K33')

        K33 = (
            Ktt - (self.rho * Kt) + (self.rho * IsserlisPow12(Kt, self.mean))
            - (self.rho * Ktdash) + (self.rho**2 * K) - (self.rho**2 * IsserlisPow12(K, self.mean))
            + (self.rho * IsserlisPow12(Ktdash, self.mean)) - (self.rho**2 * IsserlisPow12(K, self.mean)) 
            + (self.rho**2 * IsserlisPow22(K, self.mean)) 
        )

        # Sanity check
        symmetry_check(K33, matrix = 'K33')

        # whether to use u^2 observations or not
        if self.mode == 'u2':
            block_list = [K11, 
                        K21, K22, 
                        K31, K32, K33]
            
            diag_list = [K11, K22, K33]

        elif self.mode == 'only--u2':
            block_list = [K22,  
                        K32, K33]
            
            diag_list = [K22, K33]

        else:
            block_list = [K11,  
                        K31, K33]
            
            diag_list = [K11, K33]

        CoK = _symmetric_block(*block_list) 

        # Sanity check
        if self.mode == 'u2':
            assert torch.allclose(CoK, torch.cat([torch.cat([K11, K21.T, K31.T], dim = 1),
                                            torch.cat([K21, K22, K32.T], dim = 1),
                                            torch.cat([K31, K32, K33], dim = 1)]))

        # Invoking adaptive nugget
        self._nugget = torch.eye(CoK.shape[0], dtype = CoK.dtype) 

        if adaptive_nugget:
            self._nugget = _adaptive_nugget_constructor(*diag_list)
            # self._nugget = _adaptive_nugget_constructor(K11, K22, K33)
            # self._nugget = _adaptive_nugget_constructor(K11, K33)

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
                            #   [K11, K22, K33])
                            # [K11, K33])
        if self.diagnostic:
            print('Diagnostics successful !')
        return K

    # cross covariance for predicting u
    def cross_covariance(self, x, 
                         x_test, 
                         z = None):
        """ returns the H matrix for predicting u """

        # --- first row ---
        x_t, t_x = expanding_inputs(x, x_test)
        H1 = self.kernel(x_t, t_x)

        if z == None:
            printRed('provided with zero collocation points (simple Kriging H matrix)')
            return H1

        # --- second row --- 
        H2 = IsserlisPow12(H1, self.mean)
        # print(H2)

        # --- third row ---
        z_t, t_z = expanding_inputs(z, x_test)
        Ht = kernel_derivatives_1D(z_t, t_z, self.kernel,
                                   index = [1,0])
        H = self.kernel(z_t, t_z)
        H3 = Ht - (self.rho * H) + (self.rho * IsserlisPow12(H, self.mean))

        # whether to use u^2 observations or not
        if self.mode == 'u2':
            block_list = [H1, H2, H3]
        elif self.mode == 'only--u2':
            block_list = [H2, H3]
        else:
            block_list = [H1, H3]

        CoH = torch.cat(block_list, dim = 1).double()
        # CoH = torch.cat([H1, H2, H3], dim = 1).double()
        # CoH = torch.cat([H1, H3], dim = 1).double()

        return CoH

    # cross covariance for predicting u2
    def cross_covariance_u2(self, x, 
                         x_test, 
                         z = None):
        """ returns the H matrix for predicting u2 """
    
        # --- first row ---
        x_t, t_x = expanding_inputs(x, x_test)
        H = self.kernel(x_t, t_x)
        H1 = (2 * self.mean * H)
    
        if z == None:
            printRed('provided with zero collocation points (simple Kriging H matrix)')
            return H1
    
        # --- second row --- 
        H2 = IsserlisPow2(H1) + (4 * self.mean**2 * H1)
    
        # --- third row ---
        z_t, t_z = expanding_inputs(z, x_test)
        Ht = kernel_derivatives_1D(z_t, t_z, self.kernel,
                                   index = [1,0])
        H = self.kernel(z_t, t_z)
        H3 = (
                (2 * self.mean * Ht)
               - (self.rho * (2 * self.mean * H)) 
               + (self.rho * (IsserlisPow2(H) + (4 * self.mean**2 * H)))  
            )
    
        CoH = torch.cat([H1, H2, H3], dim = 1).double()
    
        return CoH

    # cross covariance for predicting du/dt - rho * u + rho * u^2
    def cross_covariance_z(self, x,
                           x_test,
                           z = None):
        """ returns the H matrix for predicting z """

        # --- first row ---
        x_t, t_x = expanding_inputs(x, x_test)
        Ht = kernel_derivatives_1D(x_t, t_x, self.kernel,
                                   index = [0,1])
        H = self.kernel(x_t, t_x)
        H1 = Ht - (self.rho * H) + (2 * self.rho * self.mean * H)

        # --- Second row ---
        Ht = kernel_derivatives_1D(x_t, t_x, self.kernel,
                                   index = [0,1])
        H = self.kernel(x_t, t_x)
        H2 = (
                (2 * self.mean * Ht)
               - (self.rho * (2 * self.mean * H)) 
               + (self.rho * (IsserlisPow2(H) + (4 * self.mean**2 * H)))  
            )

        # --- third row ---
        z_t, t_z = expanding_inputs(z, x_test)
        H = self.kernel(z_t, t_z)
        Htt = kernel_derivatives_1D(z_t, t_z, self.kernel, 
                                    index = [1,1], arg = 0)
        
        # Sanity check
        assert torch.allclose(Htt,  kernel_derivatives_1D(z_t, t_z, self.kernel, 
                                            index = [1,1], arg = 1)
                        )
        
        Ht = kernel_derivatives_1D(z_t, t_z, self.kernel, 
                                    index = [1,0])
        Htdash = kernel_derivatives_1D(z_t, t_z, self.kernel, 
                                    index = [0,1])
        
        H3 = (Htt 
                - (self.rho * (Ht + Htdash)) 
                + (2 * self.rho * self.mean * (Ht + Htdash)) 
                + (self.rho**2 * H)
                - (2 * self.rho**2 * self.mean * H)
                + (self.rho**2 * (IsserlisPow2(H) + 4 * self.mean**2 * H)) 
            )

        CoH = torch.cat([H1, H2, H3], dim = 1).double()
        
        return CoH

# Modying the centering class
# Centered observations
class LogisticCenteredValues(CenteredValues):
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
    def centering_y(self, y: Tensor, 
                    mean: float | None = None):
        """ return `torch.mean(y)` or given mean """
        y_mean = 0.0

        if not self.stationary:
            if mean is not None:
                y_mean = mean
            else: 
                y_mean = torch.mean(y) 

        return y_mean

    # Centering u^2 function
    def centering_y2(self, y: Tensor, x: Tensor, sigma = 1.0, mean = None):                       
        """ Centering the given observations Y^2. """
        
        # Centering formula -- EY^2 - (EY)^2 = k(x,x); EY^2 = k(x,x) + (EY)^2
        y_mean_sq = (self.centering_y(y, mean = mean))**2

        k = self.centering_k(x, sigma=sigma)            

        return y_mean_sq, k

    # Centering collocation points
    def centering_z(self, y: Tensor, z: Tensor, sigma = 1.0, mean = None):                       
        """ Centering the collocation points: du/dt - rho * u + rho * u^2 """

        y_mean = self.centering_y(y, mean=mean)

        term1 = (-self.rho * y_mean)
        y_mean_sq, k = self.centering_y2(y, z, sigma=sigma, mean=mean)

        term2 = (self.rho * (y_mean_sq + k))
        return term1, term2

#======================
# 1D experiment design
#======================
def Logistic_DoE(nobs = 2,                              #  
                 x_min = -10, x_max = 10,
                 x_init = 0.5,                          # x(0) initial condition 
                 rho = 1.0,                             # coefficient 
                 n_test = 200,
                 N_colloc = 100,                        # collocation points
                 random_colloc: bool = False, 
                 seed: int | None = None):

    generator = torch.Generator().manual_seed(seed) if seed is not None else 0

    train_x = torch.tensor([-7.5, -5.0, 0.0, 5.0, 7.5]).reshape(-1, 1)       # grid observations
    if seed is not None:
        train_x = x_min + (torch.rand(nobs, generator=generator) * (x_max - x_min)).reshape(-1, 1)
        train_x[0] = 0.0                                    # initial condition
    # train_x = torch.tensor([0.0]).reshape(-1, 1)

    factor = (rho * (1 - x_init) / x_init) 
    train_y = (rho /
               (rho + (torch.exp(-rho * train_x[:,0]) * factor)
            )
        )           

    test_x = torch.linspace(x_min, x_max, n_test).reshape(-1, 1)
    test_y = (rho /
                (rho + (torch.exp(-rho * test_x[:,0]) * factor)
            )
        )

    # collocation points
    train_z = torch.linspace(x_min, x_max, N_colloc).reshape(-1, 1)
    if random_colloc:
        train_z = x_min + (torch.rand(N_colloc, generator=generator) * (x_max - x_min)).reshape(-1, 1)

    # RHS of the ODE
    train_v = torch.zeros(train_z.size(dim=0))
    
    return train_x.double(), train_z.double(), train_v.double(), train_y.double(), test_x.double(), test_y.double()

#*************************
# main body
#*************************
def main(rho = 2.0,                                                             # Logistic coefficient 
        nobs: int = 0,
        N_colloc: int = 500,
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
        best_jitter: bool = False,                                              # find best jitter 
        diagnostic: bool = False,                                               # product kernel based diagnostics.
        mode: str = 'u2',                                                       # u2 observations
        mean_explicit: float | None = None,                                     # Explicit mean value for the prior (far from observations)
        seed: int | None = None,
        save: bool = False,                                                     # to save the model and figures
        **kwargs, 
    ):

    # DoE
    train_x, train_z, train_v, train_y, test_x, test_y = Logistic_DoE(rho=rho, 
                                                                      nobs = nobs if not int(nobs) == 0 else int(1), 
                                                                      N_colloc = N_colloc,
                                                                      seed=seed)               

    # choose kernel
    _kernel = choose_kernel(kernel)(lengthscale=init_lengthscale,
                                    DiffMode = True) if init_lengthscale is not None else choose_kernel(kernel)(DiffMode = True) 

    # name while saving
    file_name = f'_init{_kernel.lengthscale.tolist()}_opt{steps}_rho{rho}_colloc{N_colloc}_stat{str(stationary)}_seed{seed}'

    # define co-Kriging model
    mean = (1 - stationary) * torch.mean(train_y) if mean_explicit is None else (1 - stationary) * mean_explicit
    CoKrig = LogisticRegression(kernel = _kernel,                               # initialization co-Kriging
                                rho = rho,
                                mean = mean,
                                jitter = jitter_co_Krig,
                                true_fn = test_y,
                                mode = mode,
                                save = save,
                            )

    print(f'u_mean used for covariance computations and centering: {CoKrig.mean}')


    #================================================================
    # DIAGNOSTICS
    #================================================================
    if diagnostic:
        # Diagnostics with the product kernel
        _diagnostic_kernel = ProductKernel(lengthscale = math.pi * torch.ones(train_x.size(dim=-1)),
                                DiffMode = True)
        
        # define co-Kriging model
        stationary: bool = True
        mean = (1 - stationary) * torch.mean(train_y) if mean_explicit is None else (1 - stationary) * mean_explicit
        CoKrigDiagnostic = LogisticRegression(kernel = _diagnostic_kernel,                               # initialization co-Kriging
                                    rho = rho,
                                    mean = mean,
                                    jitter = jitter_co_Krig,
                                    true_fn = test_y,
                                    diagnostic = True
                                )

        print(f'{CoKrigDiagnostic.mean=}')

        CoKrigDiagnostic.forward(train_x, train_z)            # to check if all covariance computations are okay.
        # plt.imshow(K.matrix.detach().numpy())
        # plt.show()
    # #================================================================
    
    # #================================================================

    # Centered observations
    y_train = LogisticCenteredValues(kernel=_kernel,
                                    rho = rho,
                                    stationary=stationary
                                )
    y_mean = y_train.centering_y(train_y, mean=mean_explicit)
    y_train.forward(train_y, y_mean)

    y2_train = LogisticCenteredValues(kernel=_kernel,
                                        rho = rho,
                                        stationary=stationary
                                    )
    y_mean_sq, k = y2_train.centering_y2(train_y, train_x, mean = mean_explicit)
    y2_train.forward(train_y**2, k + y_mean_sq)

    z_train = LogisticCenteredValues(kernel=_kernel,
                                    rho = rho,
                                    stationary=stationary
                                )
    term1, term2 = z_train.centering_z(train_y, train_z, mean=mean_explicit)
    z_train.forward(train_v, term1 + term2)

    ### Concated obs
    list_obs = [y_train.centered, y2_train.centered, z_train.centered]
    red_list_obs = [y_train.centered, y2_train.centered, z_train.centered[::subsampling]]
    if mode == 'u':
        list_obs = [y_train.centered, z_train.centered]
        red_list_obs = [y_train.centered, z_train.centered[::subsampling]]
    elif mode == 'only--u2':
        list_obs = [y2_train.centered, z_train.centered]
        red_list_obs = [y2_train.centered, z_train.centered[::subsampling]]

    ConcatedObs = torch.cat(list_obs).reshape(-1).double()

    y_test = LogisticCenteredValues(kernel=_kernel,
                                    rho = rho,
                                    stationary=stationary
                                )
    
    ### predicting u^2
    k = y_test.centering_k(test_x)

    ### predicting du/dt - (rho * u) + (rho * u^2)
    term1, term2 = y_test.centering_z(train_y, test_x)

    # Training the co-Kriging model
    training_model = fit_regressor(CoKrig,
                                   steps = steps,
                                   learning_rate = learning_rate,
                                )

    ### Concated obs
    ConcatedObsReduced = torch.cat(red_list_obs).reshape(-1).double()

    training_model.central_differences_train(train_x,
                                             train_z[::subsampling],
                                             jitter = 1e-6,
                                             ConcatedObs = ConcatedObsReduced,
                                             filtering = filtering,
                                             weight_decay = weight_decay, 
                                             adaptive_nugget = adaptive_nugget,
                                        )
    training_model.plot_loss()

    # Computing the sigma
    sigma = CoKrig.LOOCVloss(train_x,
                            train_z[::subsampling], 
                            jitter = 1e-6,
                            adaptive_nugget = adaptive_nugget, 
                            ConcatedObs = ConcatedObsReduced,
                            mode = "sigma")
    
    print(f'LOOCV optimal sigma: {CoKrig.LOOCVsigma}')
    
    # Trained predictions
    y_mean_sq, k = y2_train.centering_y2(train_y, train_x, sigma=CoKrig.LOOCVsigma, mean = mean_explicit)
    y2_train.forward(train_y**2, k + y_mean_sq)

    term1, term2 = z_train.centering_z(train_y, train_z, sigma=CoKrig.LOOCVsigma, mean=mean_explicit)
    z_train.forward(train_v, term1 + term2)

    ### Recomputing concated obs
    list_obs = [y_train.centered, y2_train.centered, z_train.centered]
    red_list_obs = [y_train.centered, y2_train.centered, z_train.centered[::subsampling]]
    if mode == 'u':
        list_obs = [y_train.centered, z_train.centered]
        red_list_obs = [y_train.centered, z_train.centered[::subsampling]]
    elif mode == 'only--u2':
        list_obs = [y2_train.centered, z_train.centered]
        red_list_obs = [y2_train.centered, z_train.centered[::subsampling]]

    ConcatedObs = torch.cat(list_obs).reshape(-1).double()

    prediction, std_dev = CoKrig.predict(train_x,
                                        test_x,
                                        train_y,
                                        train_z,
                                        ConcatedObs = ConcatedObs,
                                        centering = y_train.centering,          
                                        cholesky_mode = True,
                                        adaptive_nugget = adaptive_nugget,
                                        sigma = CoKrig.LOOCVsigma,
                                        best_jitter = best_jitter,
                                    )

    # training on observations of y
    lengthscale_dict = dict(lengthscale = init_lengthscale) if init_lengthscale is not None else None
    args, kwargs = (train_x, train_y, test_x,), dict(kernel_cls = choose_gpy_kernel(kernel),       # Simple Kriging kernel
                                                    nu = choose_gpy_kernel.nu,
                                                    ard_num_dims = 1, 
                                                    # best_jitter = best_jitter                    # TO FIX
                                                ) 
    if lengthscale_dict is not None: 
        training_model.gpy_train(*args, 
                                **kwargs, 
                                **lengthscale_dict) 
    else:
        training_model.gpy_train(*args, **kwargs)

    # plotting loss curve
    training_model.plot_loss_SK()
    
    # passing the predictions directly
    training_model.pass_to_model(training_model.predictionSK,
                                 training_model.lowerSK,
                                training_model.upperSK
                            )
    
    # computing the metrics for the squared prediction
    training_model.metrics_SK(training_model.predictionSK, test_y)


    # Visualizations
    CoKrig.visualize_1D(train_x, test_x, train_y,
                        option = 'SK',
                        name = file_name,)

    CoKrig.visualize_1D(train_x,
                        test_x,
                        train_y,
                        option = 'CK',
                        # limit = True,
                        name = file_name,
                    )
    
    CoKrig.visualize_1D(train_x, test_x, train_y,
                        limit = True,
                        name = file_name,
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
    main(nobs = 5,
        rho = 1.0,
        init_lengthscale=torch.tensor([0.5]),
        filtering = [1,0,0],
        steps = 1000,
        weight_decay=0.0,
        stationary=False,
        mode = 'u2',
        kernel = 'RBF',
        save = True,
        # seed = 0,
    )