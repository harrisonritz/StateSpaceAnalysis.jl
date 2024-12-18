# StateSpaceAnalysis.jl

[![Build Status](https://github.com/harrisonritz/StateSpaceAnalysis.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/harrisonritz/StateSpaceAnalysis.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)


## Overview

*StateSpaceAnalysis.jl* is a Julia package designed for fitting linear-Gaussian state space models (SSMs) using Subspace System Identification (SSID) and Expectation Maximization (EM) algorithms. 

This package provides tools for preprocessing data, fitting models, and evaluating model performance, with methods especially tailored towards neuroimaging analysis:


### Event-related data

Neuroimaging data often has epoched/batched sequences (e.g., states x timesteps x trials). *StateSpaceAnalysis.jl* handles epoched data by reusing computations across batches, and it includes spline temporal bases for flexible input modeling over the epoch.   


### High-dimensional systems

Whole-brain modelling may require a large number of latent factors. *StateSpaceAnalysis.jl* handles scaling through efficient memory allocation, robust covariance formats (via [*PDMats.jl*](https://github.com/JuliaStats/PDMats.jl)), and regularization.  


### Data-driven initialization

We need good initialization for systems for which we don't have great domain knowledge (especially when there are many latent factors). *StateSpaceAnalysis.jl* handles parameter initialization through subspace identification methods adapted from [*ControlSystemsIdentification.jl*](https://github.com/baggepinnen/ControlSystemIdentification.jl).

This version is very preliminary, so there may be some rough edges!

## Installation

You can easily install the current release of *StateSpaceAnalysis.jl* from the Julia General Registry:

    ```julia
    using Pkg  
    Pkg.add("StateSpaceAnalysis")
    ```


You may want to work directly with the package, e.g., to modify custom functions for setting up your input bases.
You can create a local copy by cloning the github repo:

1. **Clone the repository:**
    ```sh
    git clone https://github.com/harrisonritz/StateSpaceAnalysis.jl.git
    cd StateSpaceAnalysis.jl
    ```    

2. **Open Julia in the folder and activate the package environment:**
    ```julia
    using Pkg
    Pkg.activate(".")
    Pkg.instantiate()
    ```

3. **Add the package to your Julia environment:**
    ```julia
    Pkg.add(path=".")
    using StateSpaceAnalysis
    ```

This will install all the necessary dependencies and set up the StateSpaceAnalysis.jl package for local use.

Note: You can check which directory you are working in with `pwd()` in Julia. Opening a folder in VS code sets that folder to your path. You can specify the paths in `Pkg.activate("path/to/package")` and `Pkg.add("path/to/package")` even in you aren't in the right folder.



## Walkthrough of the `example/fit_example.jl` script

### Set up `S`, the core structure which carries the parameters and data structure

```julia
S = core_struct(
    prm=param_struct(
        ... # high-level parameters
        ), 
    dat=data_struct(
        ... # data and data description
        ),
    res=results_struct(
        ... # fit metrics and model derivates
    ),
    est=estimates_struct(
        ... # scratch space
    ),
    mdl=model_struct(
        ... # estimated model parameters
    ),
    fnc=function_struct{core_struct}(
        ... # custom functions for setup
    )
);
```
This structure is used throughout the script, which allows for effective memory management (i.e., the complier can know the size of the data tensors).

### Preprocess the data:

```julia
@reset S = StateSpaceAnalysis.preprocess_fit(S);
```

Preprocessing steps within `preprocess_fit(S)`:
```julia
# read in arguments, helpful for running on a cluster
S = deepcopy(StateSpaceAnalysis.read_args(S, ARGS));

# set up the paths
StateSpaceAnalysis.setup_path(S)

# load and format the data; split for cross-validation
S = deepcopy(StateSpaceAnalysis.load_data(S));

# build the input tenors (e.g., z-score and convolve with basis)
S = deepcopy(StateSpaceAnalysis.build_inputs(S));

# transform the observed data (PCA)
S = deepcopy(StateSpaceAnalysis.project(S));

# fit baseline models to the data
StateSpaceAnalysis.null_loglik!(S);

# initialize the expectations and parameters
@reset S.est = deepcopy(set_estimates(S));
@reset S = deepcopy(generate_rand_params(S));
```

These preprocessing steps depend on custom code that you can modify.
# TODO!


### Warm-start the EM with initial parameters from Subspace Identification (SSID):

```julia
if S.prm.ssid_fit == "fit" # if fitting the SSID
    @reset S = StateSpaceAnalysis.launch_SSID(S);
elseif S.prm.ssid_fit == "load" # if loading a previously-fit SSID
    @reset S = StateSpaceAnalysis.load_SSID(S);
end
```

### Fit the parameters use EM:

```julia
@reset S = StateSpaceAnalysis.launch_EM(S);
```
The basic structure of the EM script:
```julia
for em_iter = 1:S.prm.max_iter_em

        # ==== E-STEP ================================================================
        @inline StateSpaceAnalysis.ESTEP!(S); # estimate the sufficient statistics

        # ==== M-STEP ================================================================
        @reset S.mdl = deepcopy(StateSpaceAnalysis.MSTEP(S)); # use the sufficient statistics to update the parameters

        # ==== TOTAL LOGLIK ==========================================================
        StateSpaceAnalysis.total_loglik!(S) # compute the total likelihood

        # [...] quality & convergence checks
end
```

### Save the fit:

```julia
StateSpaceAnalysis.save_results(S)
```




## Functions Overview

### `setup/custom.jl`

**This needs to be set by the user for the project-specific parameters**
- `assign_arguments`: Assigns command-line arguments to the structure.
- `select_trials`: Selects trials based on custom criteria.
- `scale_input`: Scales the input data.
- `create_input_basis`: Formats inputs with basis functions.
- `transform_observations`: Transforms observations, typically using PCA.
- `format_B_preSSID`: Formats the B matrix for SSID.
- `format_B_postSSID`: Assigns the estimated B columns to the rest of the matrix.

### `fit/launch.jl`

- `preprocess_fit`: Preprocesses the data and sets up the fitting environment.
- `launch_SSID`: Launches the SSID fitting process.
- `launch_EM`: Launches the EM fitting process.
- `load_SSID`: Loads a previously saved SSID model.
- `save_SSID`: Saves the SSID model.
- `save_results`: Saves the fitting results.

### `fit/SSID.jl`

**These function are modifed from [*ControlSystemsIdentification.jl*](https://github.com/baggepinnen/ControlSystemIdentification.jl)**
- `fit_SSID`: Performs subspace identification for state space analysis.
- `subspaceid_SSA`: modified ControlSystemsIdentification.jl for SSID

### `fit/EM.jl`

- `fit_EM`: Runs the EM algorithm for individual participants.
- `ESTEP!`: Executes the E-step of the EM algorithm.
- `MSTEP`: Executes the M-step of the EM algorithm.
- `estimate_cov!`: Estimates the latent covariance.
- `estimate_mean!`: Estimates the latent mean.
- `estimate_moments!`: update the sufficient statistics.


### `fit/posteriors.jl`

- `posterior_all`: Generates all posterior estimates (mean and covariance).
- `posterior_mean`: Generates only the posterior means.
- `posterior_sse`: Computes the sum of squared errors for the posteriors.


### `setup/setup.jl`

- `read_args`: process command line arguments (for running on the cluster)
- `setup_path`: Sets up the directory paths for saving results.
- `load_data`: Loads the data from files.
- `build_inputs`: Builds the input matrices for the model.
- `project`: projects the observations (PCA).

### `setup/generate.jl`

- `gen_rand_params`: generate random SSM parameters
- `generate_ssm_trials`: simulate trials from a set of SSM parameters 

### `setup/structs.jl`

- `param_struct`: Defines the parameters structure.
- `data_struct`: Defines the data structure.
- `results_struct`: Defines the results structure.
- `estimates_struct`: Defines the estimates structure.
- `model_struct`: Defines the model structure.
- `core_struct`: Combines all the structures into a core structure.
- `post_all`: Defines the structure for all posterior estimates.
- `post_mean`: Defines the structure for posterior means.
- `post_sse`: Defines the structure for posterior sum of squared errors.

### `utils/utils.jl`

- `tol_PD`: Ensures a matrix is positive definite with a tolerance.
- `tol_PSD`: Ensures a matrix is positive semi-definite with a tolerance.
- `demix`: Demixes the observations using the saved PCA transformation.
- `remix`: Remixes the observations using the saved PCA transformation.


