# Co-Kriging (and simple Kriging) for phyisics informed machine learning.  

import argparse
import torch

# 1D Convection equation
from problems.convection1D import main as convectionCK

# Logistic equation
from problems.logistic1D import main as logisticCK

# 1D Reaction-Diffusion equation
from problems.rxndiffusion1D import main as rxndiffusionCK

# Squared process example
from problems.squaredprocess1D import main as squaredprocessCK

#****************************
# Main body
#****************************

# string input to bool
def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

# Lengthscales are taken as characters / strings.
def _char_to_float(theta_str: list[str]) -> list[float]: 
    """ string | characters to float values. [['0', '.', '5']] --> [0.5] """
    theta_float = []
    for _i in theta_str:
        base = ''
        for _j in _i:
            base=base+_j
        theta_float.append(float(base))
    return theta_float 

# User arguments
parser = argparse.ArgumentParser(description='What do the options mean in and how to use them with main.py ?')

parser.add_argument('--problem', type=str, default='convection', choices=['convection', 'logistic', 'rd', 'squared'],
help='Choose the problem. Default = `convection`. Other options, `logistic`, `rd`, `squared`.')
parser.add_argument('--kernel', type=str, default='RBF', choices=['RBF', 'Matern32', 'Matern52'],
help='Choose the problem. Default = `RBF`. Other options, `Matern32`, `Matern52`.')

parser.add_argument('--nobs', type=int, default=0, help='Number of random solution observation locations. Default 0.')
parser.add_argument('--nc', type=int, default=900, help='Number of collocation locations. Default 900.')

parser.add_argument('--beta', type=int, default=30, help='Convection equation beta value. Default 30.')
parser.add_argument('--rho', type=float, default=1.0, help='Reaction diffusion equation (and logistic equation) rho value. Default 1.')
parser.add_argument('--nu', type=float, default=5.0, help='Diffusion nu value. Default 5.0.')

parser.add_argument('--filtering', nargs='+', type=list, default=None, help='Filter matrix for LOOCV computation.')
parser.add_argument('--iter', type=int, default=500, help='Number of optimizer steps. Default 500.')
parser.add_argument('--lr', type=float, default=0.01, help='Co-Kriging learning rate. Default 0.01.')


parser.add_argument('--nt',  type=int, default=60, help='time discretization (for space-time problems). Default 60.')
parser.add_argument('--nx',  type=int, default=60, help='space discretization (for space-time problems). Default 60.')
parser.add_argument('--jitter',  type=float, default=1e-9, help='manual jitter for numerical stability during cholesky decomposition. Default 1e-9.')
parser.add_argument('--best_jitter', type=str, default='True', help='Finite search for the best possible jitter. Default True. [OVERRIDES --JITTER]')
parser.add_argument('--random_colloc', type=str, default='True', help='Choosing uniform random collocation points. Default True.')
parser.add_argument('--init_lengthscale', nargs='+', default=None, help='The initialization of the RBF/Matern lengthscale parameters. Default [0.5] in 1D or [0.5, 0.5] in 2D')
parser.add_argument('--sigma', type=float, default=1.0, help='Explicit noise variance. Default 1.0.')

parser.add_argument('--mean', type=float, default = None, help='Explicit mean value (for non-stationary computations). Default is None')
parser.add_argument('--stationary', type=str, default='False', help='Assume mu = 0. Default False.')
parser.add_argument('--loss_landscape', type=str, default='False', help='To visualize the loss. Default False.')
parser.add_argument('--subsampling', type=int, default=2, help='subsample the collocation points for fast covariance computations during LOOCV optimization.')
parser.add_argument('--adaptive_nugget', type=str, default='True', help='Adaptive nugget for numerical stability.')

parser.add_argument('--seed', type=int, default=None, help='Random seed')
parser.add_argument('--save', type=str, default='False', help='Save model results and figures')

parser.add_argument('--paper_results', type=str, default='False', help='Settings for the paper results')

def main():
    args = parser.parse_args()
    
    # defining the variables
    steps = args.iter
    
    ## changing the strings to bool
    best_jitter = str2bool(args.best_jitter)
    random_colloc = str2bool(args.random_colloc)
    stationary = str2bool(args.stationary)
    loss_landscape = str2bool(args.loss_landscape)
    adaptive_nugget = str2bool(args.adaptive_nugget)
    save = str2bool(args.save)
    reproduce = str2bool(args.paper_results)
    
    if args.seed is not None:
        seed = args.seed
    elif args.seed is None and not args.problem == 'logistic':
        seed = 0
    else:
        seed = args.seed
     
    if args.problem in ['logistic', 'squared']:
        init_lengthscale = torch.tensor([0.5]) if args.init_lengthscale is None else torch.tensor(_char_to_float(args.init_lengthscale))
    elif args.problem in ['rd', 'convection']:
        init_lengthscale = torch.tensor([0.5, 0.5]) if args.init_lengthscale is None else torch.tensor(_char_to_float(args.init_lengthscale))
    
    filtering = _char_to_float(args.filtering) if args.filtering is not None else args.filtering
    
    kwargs = dict(Nx = args.nx, Nt = args.nt, nobs = args.nobs, N_colloc = args.nc, random_colloc = random_colloc,
                kernel = args.kernel, init_lengthscale = init_lengthscale, 
                beta = args.beta, rho = args.rho, nu = args.nu, 
                adaptive_nugget = adaptive_nugget, stationary = stationary, mean_explicit = args.mean, best_jitter = best_jitter, save = save, jitter_co_Krig = args.jitter, mode = 'u2',
                steps = steps, learning_rate = args.lr, seed = seed, weight_decay = 0.0, subsampling = args.subsampling, 
                loss_landscape = loss_landscape, sigma = args.sigma,
                sq_flag = False,
                N_initial = args.nx,
                N_boundary = args.nt, 
                filtering = filtering,
            )
    
    # problem specific 
    if args.problem == 'convection':
        fn = convectionCK
        if reproduce:                                      # For reproducibilty
            kwargs['beta'] = 30.0
            kwargs['init_lengthscale'] = torch.tensor([0.1075, 0.9328])
            kwargs['steps'] = 0
            kwargs['best_jitter'] = False

    elif args.problem == 'logistic':
        fn = logisticCK
        if reproduce:                                      # For reproducibilty
            kwargs['rho'] = 1.0
            kwargs['steps'] = 1000           
            kwargs['nobs'] = 5
            kwargs['N_colloc'] = 500
            kwargs['init_lengthscale'] = torch.tensor([0.5])
            kwargs['subsampling'] = 5
            kwargs['best_jitter'] = False

    elif args.problem == 'squared':
        fn = squaredprocessCK
        if reproduce:                                       # For reproducibilty
            kwargs['seed'] = 1                                                 
            kwargs['nobs'] = 7
            kwargs['best_jitter'] = False                              

    elif args.problem == 'rd':
        fn = rxndiffusionCK
        if reproduce:
            kwargs['rho'] = 5.0
            kwargs['nu'] = 5.0
            kwargs['init_lengthscale'] = torch.tensor([0.3, 0.3])
            kwargs['mean_explicit'] = 1.0
            kwargs['best_jitter'] = False
            kwargs['jitter_co_Krig'] = 1e-4 
            kwargs['steps'] = 0

    print(100 * '*' + '\n RESULTS WITH THE FOLLOWING SETTINGS: \n' + 100 * '*')
    for key, value in kwargs.items():
        print(f'{key} = {value}')
    print(100 * '*' + '\n')
    
    fn(**kwargs
    )

if __name__ == '__main__':
    main()
