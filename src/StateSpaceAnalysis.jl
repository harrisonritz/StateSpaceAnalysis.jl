module StateSpaceAnalysis

# import
using LinearAlgebra
using StatsFuns
using StatsBase
using Random
using Distributions
using FileIO
using MAT
using Dates
using Accessors
using MultivariateStats
using PDMats
using SpecialFunctions
using Serialization
using ControlSystems
using ControlSystemIdentification
using BSplines
using OffsetArrays


try
    using MATLAB
catch
    println("NO MATLAB; resolve or set S.prm.write_mat=false")
end



# custom functions
include("setup/custom.jl")
export assign_arguements, select_trials, scale_input, create_input_basis, launch_EM, load_SSID


# fit functions
include("fit/wrapper.jl")
export preprocess_fit, launch_SSID, launch_EM, load_SSID, save_SSID, save_results

include("fit/EM.jl")
export fit_EM, ESTEP!, MSTEP, estimate_cov!, filter_cov!, filter_cov_KF!, smooth_cov!,
    estimate_mean!, filter_mean!, filter_mean_KF!, smooth_mean!, init_moments!, estimate_moments!,
    total_loglik!, total_loglik, test_loglik!, test_loglik, test_orig_loglik, null_loglik!

include("fit/SSID.jl")

include("fit/likelihoods.jl")
export ll_R2, log_post_v0, log_post, init_lik, dyn_lik, obs_lik, 
    total_loglik!, total_loglik, 
    test_loglik!, test_loglik, 
    test_orig_loglik, null_loglik!

include("fit/posteriors.jl")
export posterior_all, posterior_mean, posterior_sse


# setup functions
include("setup/setup.jl")
export  read_args, setup_path, load_data, build_inputs, whiten, save_results

include("setup/structs.jl")
export  core_struct, param_struct, data_struct, results_struct, estimates_struct, set_estimates, model_struct,
        set_model, transform_model

include("setup/generate.jl")
export generate_rand_params, generate_ssm_trials


# utility functions
include("utils/utils.jl")
export zsel, zsel_tall, zdim, init_PD, tol_PD, init_PSD, tol_PSD, diag_PD, format_noise, sumsqr, split_list, demix, remix
export format_B_preSSID, format_B_postSSID
export report_R2, report_params



# TODO: integrate plots
# include("utils/plots.jl")




end
