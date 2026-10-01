
# imports \
import sys
import torch

sys.path.append('..')        # To search in the right directory 

from physics import PhysicsRegression, PhysicsMatrix, _adaptive_nugget_constructor, CenteredValues, fit_regressor
from kernels import expanding_inputs, IsserlisPow2, symmetry_check, dim_check
from kernels import choose_kernel
from SimpleKrigGpytorch import choose_gpy_kernel
from torch import Tensor
from typing import Literal

import math

# modifying the forward function and the cross-covariance functions
class SquaredRegression(PhysicsRegression):
    r"""
    Squared process co-Kriging class. 
    
    Attributes\:
    
    Define the following methods:
        1. `.forward()`: this is going to give the observation covariance matrix.
        2. `.cross_covariance(mode:str='u')`: define one or multiple modes. 
    """
    def __init__(
        self,
        mean: float = 0.0,
        *args, 
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.mean = float(mean)

    def forward(self, x, 
                z: Tensor | None = None,                                # Dummy variable
                adaptive_nugget = False, 
                diag_blocks = True)-> PhysicsMatrix:
        """ returns the K matrix """

        xsymm = expanding_inputs(x)
        CoK = (
            IsserlisPow2(self.kernel(xsymm)) 
            + (4 * self.mean**2 * self.kernel(xsymm))
        )  

        # Invoking adaptive nugget
        self._nugget = torch.eye(CoK.shape[0], dtype = CoK.dtype) 
        if adaptive_nugget:
            self._nugget = _adaptive_nugget_constructor(CoK)

        # Sanity check
        symmetry_check(CoK)

        # Sanity check
        dim_check(CoK, self._nugget)

        # Defining a physicsmatrix object
        K = PhysicsMatrix(CoK, self._nugget)
        if diag_blocks:
            K = PhysicsMatrix(CoK, 
                              self._nugget, 
                              [CoK])
        return K

    def cross_covariance(self,
                    x: Tensor,
                    x_test: Tensor,
                    z: Tensor | None = None,                             # Dummy variable
                    # mu: float = 0.0,
            )-> Tensor:
        """ returns the H matrix """
    
        x_t, t_x = expanding_inputs(x, x_test)
        H = (IsserlisPow2(self.kernel(x_t, t_x)) 
            #  + (4 * self.mean * self.kernel(x_t, t_x))  
             + (4 * self.mean**2 * self.kernel(x_t, t_x))
        )

        CoH = torch.cat([H,], dim = 1).double()
    
        return CoH

# Modying the centering class
# Centered observations
class SqCenteredValues(CenteredValues):
    """ Centering functions and Tensor that stores all of the centered information (the centering, the centered output, etc.) """
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

    # Centering function
    def centering_y2(self, y: Tensor, x: Tensor, 
                     stationary: bool = True):                      # Stationary assumption for Y 
        """ Centering the given observations Y^2. """
        # Storing original
        original_val = self.kernel.diag_mode 
        self.kernel.diag_mode = True
        
        # Slightly faster -- Centering formula -- EY^2 - (EY)^2 = k(x,x); EY^2 = k(x,x) + (EY)^2
        y_mean_sq = 0.0
        if not stationary:
            y_mean_sq = (torch.mean(y))**2 
        k = self.kernel(x).squeeze()            # For 1D problems
        
        self.kernel.diag_mode = original_val
        return y_mean_sq, k

    # kernel based centering
    def centering_k(self, x: Tensor):
        # Storing original
        original_val = self.kernel.diag_mode 
        self.kernel.diag_mode = True

        k = self.kernel(x).squeeze()            # For 1D problems

        self.kernel.diag_mode = original_val
        return k


#======================
# 1D experiment design
#======================
def Sq_1D_DoE(nobs = 50, x_min = 0.0, x_max = 3*math.pi,
            n_test = 200, seed = 1):

    generator = torch.Generator().manual_seed(seed)

    train_x = x_min + (torch.rand(nobs, generator=generator) * (x_max - x_min)).reshape(-1, 1) 
    train_y =  torch.sin(train_x[:, 0])         # torch.exp(-0.1 * train_x[:, 0]) *
    train_y2 = torch.square(train_y)

    test_x = torch.linspace(x_min, x_max, n_test).reshape(-1, 1)
    test_y2 =torch.square(torch.sin(test_x[:, 0]))  # torch.exp(-0.1 * test_x[:, 0])
    
    return train_x.double(), train_y.double(), train_y2.double(), test_x.double(), test_y2.double()

#***************************
# main body 
#***************************
def main(nobs: int = 5,
        kernel: Literal['RBF', 'Matern32', 'Matern52'] = 'RBF', 
        init_lengthscale: float | Tensor | None = None, 
        stationary: bool = True,                                                # stationary assumption
        jitter_co_Krig: float = 1e-6,                                           # jitter for numerical stability
        adaptive_nugget: bool = False,                                           # adaptive nugget option
        weight_decay: float = 1e-4,                                             # optimizer weight decay
        steps: int = 0,                                                         # optimizer steps
        learning_rate: float = 0.01,                                            # optimizer learning rate 
        best_jitter: bool = False,                                              # find best jitter 
        sq_flag: bool = True,                                                   # squaring simple Kriging predictions OR simple Kriging the squared process
        seed: int = 0, 
        save: bool = False,                                                     # saving results
        **kwargs,
    ):

    # DoE
    train_x, train_y, train_y2, test_x, test_y2 = Sq_1D_DoE(nobs=nobs, seed=seed)

    # choose kernel
    _kernel = choose_kernel(kernel)(lengthscale=init_lengthscale,
                                        DiffMode = True) if init_lengthscale is not None else choose_kernel(kernel)(DiffMode = True) 

    # name while saving
    file_name = f'_init{_kernel.lengthscale.tolist()}_opt{steps}'

    # define co-Kriging model
    mean = (1 - stationary) * torch.mean(train_y)
    CoKrig = SquaredRegression(kernel = _kernel,
                               true_fn = test_y2,
                               jitter = jitter_co_Krig,
                               mean = mean,
                               save = save,
                            )

    print(f'u_mean used for covariance computations and centering: {CoKrig.mean}')

    # Centered observations
    y2_train = SqCenteredValues(kernel=_kernel)
    y_mean_sq, kx = y2_train.centering_y2(train_y, train_x, 
                                          stationary=stationary)
    y2_train.forward(train_y2, y_mean_sq + kx)

    y2_test = SqCenteredValues(kernel = _kernel)
    kx = y2_test.centering_k(test_x)
    y2_test.centering = kx + y_mean_sq

    # train the co-Kriging model
    training_model = fit_regressor(CoKrig, 
                                    steps = steps,
                                    learning_rate=learning_rate)
    
    training_model.central_differences_train(train_x,
                                             jitter = 1e-6,
                                             ConcatedObs = y2_train.centered,
                                             weight_decay = weight_decay, 
                                             adaptive_nugget = adaptive_nugget,
                                        )
    training_model.plot_loss()

    #=================================
    # Squaring the simple Kriging predictions
    #=================================
    if sq_flag:
        # training on observations of y
        lengthscale_dict = dict(lengthscale = init_lengthscale) if init_lengthscale is not None else None
        args, kwargs = (train_x, train_y, test_x,), dict(kernel_cls = choose_gpy_kernel(kernel),       # Simple Kriging kernel
                                                        nu = choose_gpy_kernel.nu,
                                                        ard_num_dims = 1, 
                                                        # best_jitter = best_jitter                     # TO FIX
                                                    ) 

        if lengthscale_dict is not None: 
            training_model.gpy_train(*args, 
                                    **kwargs, 
                                    **lengthscale_dict) 
        else:
            training_model.gpy_train(*args, **kwargs)

        # plotting loss curve
        training_model.plot_loss_SK()

        # computing the 2sigma intervals        
        foursigma = training_model.upperSK - training_model.lowerSK             # Simple Kriging confidence intervals.

        # passing the squared predictions
        training_model.pass_to_model(training_model.predictionSK**2,
                                     training_model.predictionSK**2 - (foursigma / 2),
                                    training_model.predictionSK**2 + (foursigma / 2)
                                )
        
        # computing the metrics for the squared prediction
        training_model.metrics_SK(training_model.predictionSK**2, test_y2)  

    #=================================
    # Simple Kriging on the squared process
    #=================================
    else:
        # training on observations of y2
        lengthscale_dict = dict(lengthscale = init_lengthscale) if init_lengthscale is not None else None
        args, kwargs = (train_x, train_y2, test_x,), dict(kernel_cls = choose_gpy_kernel(kernel),       # Simple Kriging kernel
                                                        nu = choose_gpy_kernel.nu,
                                                        ard_num_dims = 1, 
                                                        # best_jitter = best_jitter                     # TO FIX
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
        training_model.metrics_SK(training_model.predictionSK, test_y2)   


    # Computing the sigma
    print(f'LOOCV optimal sigma: {CoKrig.LOOCVloss(train_x, 
                                                jitter = 1e-6,
                                                adaptive_nugget = adaptive_nugget, 
                                                ConcatedObs = y2_train.centered,
                                                mode = 'sigma').item()}')

    # Trained predictions
    prediction, std_dev = CoKrig.predict(train_x,
                                        test_x,
                                        train_y,
                                        ConcatedObs=y2_train.centered,
                                        centering=y2_test.centering,
                                        cholesky_mode=True,
                                        adaptive_nugget=adaptive_nugget,
                                        sigma = CoKrig.LOOCVsigma,
                                        best_jitter = best_jitter,
                                        min_val=0.1                         # to make conditional variance positive 
                                    )
    
    CoKrig.visualize_1D(train_x, test_x, train_y2,
                        option = 'SK',
                        name = file_name,)
    CoKrig.visualize_1D(train_x, test_x, train_y2,
                        option = 'CK',
                        name = file_name)
    CoKrig.visualize_1D(train_x, test_x, train_y2,
                        # limit = True,
                        name = file_name)

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
    main(nobs = 7,
        kernel='RBF',
        seed = 1,
        steps = 500,
        init_lengthscale=torch.tensor([0.5]),
        stationary = False,
        sq_flag = False,
        save = True,
    )





