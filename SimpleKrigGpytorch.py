# Simple Krigng via gpytorch. NLL Loss and LOOCV loss from the gpytorch library.
import gpytorch as gpt
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Literal
from torch import Tensor
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

#===============
# Colored text
#===============
RED, END = '\033[91m', '\033[0m'
printRed = lambda sTxt: print(RED + sTxt + END)

def _gpy_flash():
    printRed(100*'=')
    print('Simple Kriging as GPR from gpytorch')
    printRed(100*'=')

def choose_gpy_kernel(kernel: Literal['RBF', 'Matern32', 'Matern52'] = 'RBF'):
    choose_gpy_kernel.nu = 2.5
    if kernel == 'RBF':
        return gpt.kernels.RBFKernel
    elif kernel == 'Matern32':
        choose_gpy_kernel.nu = 1.5
        return gpt.kernels.MaternKernel
    elif kernel == 'Matern52':
        return gpt.kernels.MaternKernel
    else:
        raise ValueError('Unsupported kernel choice')
    

class MyLeaveOneOutPseudoLikelihood(ExactMarginalLogLikelihood):
    r"""
    The leave one out cross-validation (LOO-CV) likelihood from RW 5.4.2 for an exact Gaussian process with a
    Gaussian likelihood. This offers an alternative to the exact marginal log likelihood where we
    instead maximize the sum of the leave one out log probabilities :math:`\log p(y_i | X, y_{-i}, \theta)`.
    
    Naively, this will be O(n^4) with Cholesky as we need to compute `n` Cholesky factorizations. Fortunately,
    given the Cholesky factorization of the full kernel matrix (without any points removed), we can compute
    both the mean and variance of each removed point via a bordered system formulation making the total
    complexity O(n^3).
 
    This is the modified version of the original code (gpytorch.mlls.LeaveOneOutPseudoLikelihood(),  
    https://docs.gpytorch.ai/en/latest/marginal_log_likelihoods.html#leaveoneoutpseudolikelihood) 
    computes the likelkihood which is given as: 

    `-1/2 \log(\sigma_i^2) - (y_i - \mu_i)^2 / 2\sigma_i^2 - 1/2 \log(2 \pi)`  

    We single out the `(y_i - \mu_i)^2` part.

    The LOO-CV approach can be more robust against model mis-specification as it gives an estimate for the
    (log) predictive probability, whether or not the assumptions of the model is fulfilled.

    .. note::
        This module will not work with anything other than a :obj:`~gpytorch.likelihoods.GaussianLikelihood`
        and a :obj:`~gpytorch.models.ExactGP`. It also cannot be used in conjunction with
        stochastic optimization.

    :param ~gpytorch.likelihoods.GaussianLikelihood likelihood: The Gaussian likelihood for the model
    :param ~gpytorch.models.ExactGP model: The exact GP model

    Example:

        >>> # model is a gpytorch.models.ExactGP
        >>> # likelihood is a gpytorch.likelihoods.Likelihood
        >>> loocv = gpytorch.mlls.LeaveOneOutPseudoLikelihood(likelihood, model)
        >>>
        >>> output = model(train_x)
        >>> loss = -loocv(output, train_y)
        >>> loss.backward()
    """

    def __init__(self, likelihood, model):
        super().__init__(likelihood=likelihood, model=model)
        self.likelihood = likelihood
        self.model = model
        # self.model.covar_module.outputscale = torch.tensor([1.0])
    
    def forward(self, function_dist: MultivariateNormal, target: Tensor, *params) -> Tensor:
            r"""
            Computes the leave one out likelihood given :math:`p(\mathbf f)` and :math:`\mathbf y`
    
            :param ~gpytorch.distributions.MultivariateNormal output: the outputs of the latent function
                (the :obj:`~gpytorch.models.GP`)
            :param torch.Tensor target: :math:`\mathbf y` The target values
            :param dict kwargs: Additional arguments to pass to the likelihood's forward function.
            """
            output = self.likelihood(function_dist, *params)
            m, L = output.mean, output.lazy_covariance_matrix.cholesky(upper=False)
            m = m.reshape(*target.shape)
            # print(m)    <<<< # Troubleshooting
            identity = torch.eye(*L.shape[-2:], dtype=m.dtype, device=m.device)
    
            sigma2 = 1.0 / L._cholesky_solve(identity, upper=False).diagonal(dim1=-1, dim2=-2)  # 1 / diag(inv(K))
            # print(sigma2)   <<<< # Troubleshooting
    
            mu = target - L._cholesky_solve((target - m).unsqueeze(-1), upper=False).squeeze(-1) * sigma2
            #********************
            # [REMOVED]
            # term1 = -0.5 * sigma2.log()
            #********************
            term2 = -0.5 * (target - mu).pow(2.0) # / sigma2    
            res = ( # term1 +                      ^^^^^^^^
                    # ^^^^^^^                      [REMOVED]
                    #[REMOVED]
                 term2
                ).sum(dim=-1)
    
            res = self._add_other_terms(res, params)
    
            # Scale by the amount of data we have and then add on the scaled constant
            num_data = target.size(-1)
            return res.div_(num_data) # - 0.5 * math.log(2 * math.pi)
                                      #    ^^^^^^^^^^^^^^^^^^^^^^^^^
                                      #           [REMOVED]

# Define the Exact GP Model (Simple Kriging)
class ExactGPModel(gpt.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, 
                 *args,
                 kernel_cls = gpt.kernels.RBFKernel, 
                 **kwargs):

        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        
        # self.mean_module = gpt.means.ConstantMean()
        self.mean_module = gpt.means.ZeroMean()

        # choice of covariance function with problem dimension
        base_kernel = kernel_cls(*args, **kwargs)

        # Pass it to ScaleKernel (noise variance)
        self.covar_module = gpt.kernels.ScaleKernel(base_kernel)
        # self.covar_module = base_kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpt.distributions.MultivariateNormal(mean_x, covar_x)
        
# Optimizer function
def fit_regressor(x_train,
                  y_train, 
                  x_test,
                  *args,
                  _loss: MyLeaveOneOutPseudoLikelihood = MyLeaveOneOutPseudoLikelihood, 
                  lr: float = 0.01,
                  iter: int = 500,
                  interactive: bool = False,
                  best_jitter: bool = False,
                  u_true: Tensor | None = None,                  # `best_jitter` requires `u_true`.
                  _jitter: float = 1e-6,
                  kernel_cls = gpt.kernels.RBFKernel,
                  **kwargs) -> tuple[Tensor, Tensor, Tensor]:

    # Initialize likelihood and model
    likelihood = gpt.likelihoods.GaussianLikelihood()

    # likelihood = gpt.likelihoods.FixedNoiseGaussianLikelihood(noise=torch.ones(x_train.size(dim=0)),
    #                                                           learn_additional_noise=False)
    
    # Fixing the noise covariance to ensure gpytorch model is interpolating 
    likelihood.noise_covar.initialize(noise=1e-4)
    likelihood.raw_noise.requires_grad_(False)
    
    model = ExactGPModel(x_train, y_train, likelihood, *args, 
                            kernel_cls = kernel_cls, **kwargs)

    # Setup the LOOCV Pseudo-Likelihood Loss Objective
    # This replaces gpytorch.mlls.ExactMarginalLogLikelihood with LOOCV
    mll = _loss(likelihood, model)      

    # Training Loop (Optimization)
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []

    _gpy_flash()
    print(f'Using gpytorch kernel {model.covar_module.base_kernel} with loss {_loss}')
    #=============================================
    # [FOR A MORE DETAILED INFO.]
    # for name, param in model.named_parameters(): 
    #     # if "lengthscale" in name: 
    #         print(f'Before training {name} : {param.data}')
    #=============================================
    if 'lengthscale' in kwargs:
        initial_lengthscale = kwargs.get('lengthscale')
        model.covar_module.base_kernel.lengthscale = initial_lengthscale
    print(f'Before training lengthscale : {model.covar_module.base_kernel.lengthscale}')
    # print(f'Before training lengthscale : {model.covar_module.lengthscale}')

    for i in tqdm(range(iter), colour = 'green', position=0):
        
        optimizer.zero_grad()
        
        # Forward pass: compute the output distribution from the model
        output = model(x_train)
        
        # Compute the negative LOOCV pseudo-likelihood loss
        loss = -mll(output, y_train)

        if i == 0:
            print(f'Kriging {_loss} loss pre-training: {loss.detach().item()}')

        if interactive:
            print(f"Iter {i+1}/{iter} - Loss: {loss.item():.3f}")

        if i == iter - 1:
            print(f'Kriging {_loss} loss post-training: {loss.detach().item()}')

        loss.backward()
        losses.append(loss.detach().item())
        optimizer.step()

    # Making Predictions (Kriging Interpolation)
    model.eval()
    likelihood.eval()
    # for name, param in model.named_parameters(): 
    # #     # if "lengthscale" in name: 
    #         print(f'After training {name} : {param.data}')
    print(f'After training lengthscale : {model.covar_module.base_kernel.lengthscale}')
    # print(f'After training lengthscale : {model.covar_module.lengthscale}')
    
    # ONLY for 2D problems 
    with torch.no_grad(), gpt.settings.fast_pred_var():
        if best_jitter:
            best_score = 1e10
            print('Searching for the best jitter value ...')
            for _jitter in [0.5, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14]:
                with gpt.settings.cholesky_jitter(_jitter):
                    observed_pred = likelihood(model(x_test))

                    rmse = torch.mean((
                                        observed_pred.mean.reshape(u_true.size(dim=0),              # this part is for 2D problems only
                                                        u_true.size(dim=1)) - u_true
                                    )**2
                                )**0.5
                    if rmse < best_score:
                        best_score = rmse
                        bestjitter = _jitter
                        bestprediction = observed_pred
            
            print(f'Best jitter found = {bestjitter} with RMSE = {best_score}')
            prediction = bestprediction

        elif not best_jitter:
            with gpt.settings.cholesky_jitter(_jitter):
                prediction = likelihood(model(x_test))
            
    return losses, prediction.mean, prediction.confidence_region()
    

#*********************************
# Main body (Optional)
#*********************************
def main():
    # Generate some synthetic 1D data (Kriging input locations and values)
    torch.manual_seed(42)
    train_x = torch.linspace(0, 1, 25)
    train_y = torch.sin(train_x * (2 * 3.14159)) + 0.1 * torch.randn_like(train_x)
    test_x = torch.linspace(0, 1, 200)

    # Comparing LOOCV loss for two choices of ScaleKernel noise_var
    likelihood = gpt.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood,)
    output = model(train_x)

    loocv = MyLeaveOneOutPseudoLikelihood(likelihood, model)


    model.covar_module.outputscale = torch.tensor([0.1])
    model.covar_module.base_kernel.lengthscale = torch.tensor([0.3])
    print(f'model noise variance = {model.covar_module.outputscale}')
    print(f'model lengthscale = {model.covar_module.base_kernel.lengthscale}')
    print(f'For noise var = {model.covar_module.outputscale}, LOOCV loss: {-loocv(output, train_y)}')

    model = ExactGPModel(train_x, train_y, likelihood,)
    output = model(train_x)
    
    loocv = MyLeaveOneOutPseudoLikelihood(likelihood, model)

    model.covar_module.outputscale = torch.tensor([42.0])
    model.covar_module.base_kernel.lengthscale = torch.tensor([0.3])
    print(f'model noise variance = {model.covar_module.outputscale}')
    print(f'model lengthscale = {model.covar_module.base_kernel.lengthscale}')
    print(f'For noise var = {model.covar_module.outputscale}, LOOCV loss: {-loocv(output, train_y)}')
        
    # Predictions and uncertainties
    losses, mean, (lower, upper) = fit_regressor(train_x, train_y, test_x, 
                                                #  _loss=gpt.mlls.LeaveOneOutPseudoLikelihood,
                                                _loss = MyLeaveOneOutPseudoLikelihood,
                                                 kernel_cls=gpt.kernels.MaternKernel,
                                                 iter = 100,
                                                 nu = 2.5,
                                                 ard_num_dims = 1)
    
    # visualize
    train_x_plot = train_x.cpu()
    train_y_plot = train_y.cpu()
    test_x_plot = test_x.cpu()
    mean_plot = mean.cpu()
    lower_plot, upper_plot = lower.cpu(), upper.cpu()
    
    plt.plot(test_x_plot, torch.sin(test_x * (2 * 3.14159)).cpu(), linestyle = '--', color = 'black')
    plt.plot(test_x_plot, mean_plot, 'blue')
    plt.scatter(train_x_plot, train_y_plot, marker = 'X', s = 100, color = 'black')
    plt.fill_between(test_x_plot, lower_plot, upper_plot, color = 'blue', alpha = 0.25)
    plt.show()
    
    print("\nTraining complete! Kriging model optimized using LOOCV loss.")

if __name__ == "__main__":
    main()