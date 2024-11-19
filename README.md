# StateSpaceAnalysis.jl

[![Build Status](https://github.com/harrisonritz/StateSpaceAnalysis.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/harrisonritz/StateSpaceAnalysis.jl/actions/workflows/CI.yml?query=branch%3Amain)


## Overview

StateSpaceAnalysis.jl is a Julia package designed for fitting linear Gaussian state space models (lg-SSMs) using Subspace System Identification (SSID) and Expectation Maximization (EM) algorithms. This package provides tools for preprocessing data, fitting models, and evaluating model performance.

This version is very preliminary, so there may be some rough edges!

## Installation

To install the StateSpaceAnalysis.jl package, follow these steps:

1. **Clone the repository:**
    ```sh
    git clone https://github.com/harrisonritz/StateSpaceAnalysis.jl.git
    cd StateSpaceAnalysis.jl
    ```

2. **Open Julia and activate the package environment:**
    ```julia
    using Pkg
    Pkg.activate(".")
    Pkg.instantiate()
    ```

3. **Add the package to your Julia environment:**
    ```julia
    Pkg.add(path=".")
    ```

This will install all the necessary dependencies and set up the StateSpaceAnalysis.jl package for use.



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
**These function are modifed from the excellent ControlSystemsIdentification.jl package**
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
- `read_args`: process command line arguements (for running on the cluster)
- `setup_path`: Sets up the directory paths for saving results.
- `load_data`: Loads the data from files.
- `build_inputs`: Builds the input matrices for the model.
- `whiten`: Whitens the observations (PCA).

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
- `zsel`: Standardizes the selected elements of a vector.
- `zsel_tall`: Standardizes the selected elements of a tall vector.
- `zdim`: Standardizes the elements of an array along specified dimensions.
- `init_PD`: Initializes a positive definite matrix.
- `tol_PD`: Ensures a matrix is positive definite with a tolerance.
- `init_PSD`: Initializes a positive semi-definite matrix.
- `tol_PSD`: Ensures a matrix is positive semi-definite with a tolerance.
- `diag_PD`: Creates a diagonal positive definite matrix.
- `format_noise`: Formats noise matrices based on the specified type.
- `sumsqr`: Computes the sum of squares of a vector.
- `split_list`: Splits a string by a delimiter.
- `demix`: Demixes the observations using the saved PCA transformation.
- `remix`: Remixes the observations using the saved PCA transformation.








## Running the Example
To run the example fitting script, follow these steps:

1. Set the paths in `example/fit_example.jl`:
    ```julia
    run_cluster = length(ARGS)!=0;
    if run_cluster
        src_path = "/home/hr0283/HallM_StateSpaceAnalysis/src"
        save_path = "/scratch/gpfs/hr0283/HallM_StateSpaceAnalysis/src";
    else 
        src_path =  "/Users/hr0283/Projects/StateSpaceAnalysis.jl/src"
        save_path = "/Users/hr0283/Projects/StateSpaceAnalysis.jl/example";
    end
    ```

2. Load the necessary packages and configure the system:
    ```julia
    using StateSpaceAnalysis
    using Accessors
    using Random
    using LinearAlgebra
    using Dates
    using Revise
    ```

3. Set the parameters and data structure:
    ```julia
    S = core_struct(
        prm=param_struct(
            seed = rand_seed,
            model_name = "test",
            changelog = "run test",
            load_name = "HallMcMaster2019_ITI100-Cue200-ISI400-Trial200_srate@125_filt@0-30",
            load_path = "/Users/hr0283/Projects/StateSpaceAnalysis.jl/example/example-data",
            pt_list = 1:1,
            max_iter_em = 500,
            ssid_fit = "fit",
            ssid_save = false,
            ssid_type = :CVA,
            ssid_lag = 24,
        ),
        dat=data_struct(
            sel_event = 2:4,
            pt = 1,
            x_dim = 24,
            basis_name = "bspline",
            spline_gap = 5,
        ),
        res=results_struct(),
        est=estimates_struct(),
        mdl=model_struct(),
    );
    ```

4. Run the fitting process:
    ```julia
    @reset S.res.startTime_all = Dates.format(now(), "mm/dd/yyyy HH:MM:SS");
    println("Starting fit at $(S.res.startTime_all)")

    @reset S = StateSpaceAnalysis.preprocess_fit(S);

    if S.prm.ssid_fit == "fit"
        @reset S = StateSpaceAnalysis.launch_SSID(S);
    elseif S.prm.ssid_fit == "load"
        @reset S = StateSpaceAnalysis.load_SSID(S);
    end

    @reset S = StateSpaceAnalysis.launch_EM(S);

    @reset S.res.endTime_all = Dates.format(now(), "mm/dd/yyyy HH:MM:SS");
    println("Finished fit at $(Dates.format(now(), "mm/dd/yyyy HH:MM:SS"))")
    ```

5. Optionally, plot diagnostics and save the fit:
    ```julia
    do_plots = false
    if do_plots
        try
            StateSpaceAnalysis.plot_loglik_traces(S)
            StateSpaceAnalysis.plot_avg_pred(S)
            StateSpaceAnalysis.plot_params(S)
        catch
        end
    end

    if S.prm.do_save
        println("\n========== SAVING FIT ==========")
        StateSpaceAnalysis.save_results(S)
    end
    ```

This will run the example fitting script, performing SSID and EM fitting on the provided data.


