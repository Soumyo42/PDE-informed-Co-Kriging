# Learning Physical Fields with PDE-Informed Gaussian Processes: A Frugal Alternative to PINNs

Description: Co-Kriging for solving and learning physical fields from sparse observations and PDE constraints.

This is the accompanying python code of the paper.

The supported problems:

- 1D Convection Equation
- Logistic Equation
- 1D Reaction-Diffusion Equation
- Squared Process 

The code supports RBF and Matérn kernels, LOOCV-based hyperparameter selection and adaptive numerical stabilization.

A two-dimensional Darcy-flow example (`Darcy_jupyter_example/`)  is also provided as a set of Jupyter notebooks. This is a `numpy` implementation.  

---

## Repository Structure

```text
.
├── Darcy_jupyter_example
│   ├── 2DDarcyCoKrig.ipynb
│   ├── 2DDarcyFlow.ipynb
│   └── DarcySoln.txt
├── kernels.py
├── main.py
├── physics.py
├── problems
│   ├── convection1D.py
│   ├── logistic1D.py
│   ├── rxndiffusion1D.py
│   └── squaredprocess1D.py
├── SimpleKrigGpytorch.py
└── requirements.txt
```
## Installation
1. Clone the Repository

```bash
cd <repository>
git clone https://github.com/Soumyo42/PDE-informed-Co-Kriging.git
```

2. Create a Virtual Environment
Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```
Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Activate the virtual environment
Linux / macOS:
```bash
source venv/bin/activate 
```
Windows Command Prompt (CMD)
```bash
venv\Scripts\activate.bat
```

5. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Instructions on how to use `main.py`

Run,
```bash
python main.py --help
```

```bash
usage: main.py [-h] [--problem {convection,logistic,rd,squared}] [--kernel {RBF,Matern32,Matern52}] [--nobs NOBS] [--nc NC] [--beta BETA] [--rho RHO] [--nu NU]
               [--filtering FILTERING [FILTERING ...]] [--iter ITER] [--lr LR] [--nt NT] [--nx NX] [--jitter JITTER] [--best_jitter BEST_JITTER]
               [--random_colloc RANDOM_COLLOC] [--init_lengthscale INIT_LENGTHSCALE [INIT_LENGTHSCALE ...]] [--stationary STATIONARY] [--loss_landscape LOSS_LANDSCAPE]
               [--subsampling SUBSAMPLING] [--adaptive_nugget ADAPTIVE_NUGGET] [--seed SEED] [--save SAVE] [--paper_results PAPER_RESULTS]

What do the options mean in and how to use them with main.py ?

options:
  -h, --help            show this help message and exit
  --problem {convection,logistic,rd,squared}
                        Choose the problem. Default = `convection`. Other options, `logistic`, `rd`, `squared`.
  --kernel {RBF,Matern32,Matern52}
                        Choose the problem. Default = `RBF`. Other options, `Matern32`, `Matern52`.
  --nobs NOBS           Number of random solution observation locations. Default 0.
  --nc NC               Number of collocation locations. Default 900.
  --beta BETA           Convection equation beta value. Default 30.
  --rho RHO             Reaction diffusion equation (and logistic equation) rho value. Default 1.
  --nu NU               Diffusion nu value. Default 5.0.
  --filtering FILTERING [FILTERING ...]
                        Filter matrix for LOOCV computation.
  --iter ITER           Number of optimizer steps. Default 500.
  --lr LR               Co-Kriging learning rate. Default 0.01.
  --nt NT               time discretization (for space-time problems). Default 60.
  --nx NX               space discretization (for space-time problems). Default 60.
  --jitter JITTER       manual jitter for numerical stability during cholesky decomposition. Default 1e-9.
  --best_jitter BEST_JITTER
                        Finite search for the best possible jitter. Default True. [OVERRIDES --JITTER]
  --random_colloc RANDOM_COLLOC
                        Choosing uniform random collocation points. Default True.
  --init_lengthscale INIT_LENGTHSCALE [INIT_LENGTHSCALE ...]
                        The initialization of the RBF/Matern lengthscale parameters. Default [0.5] in 1D or [0.5, 0.5] in 2D
  --stationary STATIONARY
                        Assume mu = 0. Default False.
  --loss_landscape LOSS_LANDSCAPE
                        To visualize the loss. Default False.
  --subsampling SUBSAMPLING
                        subsample the collocation points for fast covariance computations during LOOCV optimization.
  --adaptive_nugget ADAPTIVE_NUGGET
                        Adaptive nugget for numerical stability.
  --seed SEED           Random seed
  --save SAVE           Save model results and figures
  --paper_results PAPER_RESULTS
                        Settings for the paper results
```

### To reproduce the paper results
 
```bash
python main.py --problem {convection,logistic,rd,squared} --paper_results True
```

## Darcy Flow example

`2DDarcyFlow.ipynb` is a finite-differences solver (`numpy`-based) for the 2D Darcy flow (as presented in https://arxiv.org/pdf/2010.08895):

$$
\begin{equation}
\nabla \cdot (a(x) \nabla u(x)) = f(x) \quad \text{in $\Omega$}  
\end{equation} 
$$

$$
\begin{equation}
u(x) = 0 \quad \text{in $\partial\Omega$}  
\end{equation} 
$$

We used a determinstic, diffusion coefficient, $a(x, y) \coloneqq e^{-x} sin(y)$.

The solution is saved as  `DarcySoln.txt` which is then loaded for use in `2DDarcyCoKrig.ipynb`. 
