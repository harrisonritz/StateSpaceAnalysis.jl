
# SET PATHS =============================================================
# EDIT THIS!
run_cluster = length(ARGS)!=0;
if run_cluster
    root_path = "/home/hr0283/HallM_StateSpaceAnalysis/src"
    save_path = "/scratch/gpfs/hr0283/HallM_StateSpaceAnalysis/src";
else 
    root_path =  "/Users/hr0283/Projects/StateSpaceAnalysis.jl/src"
    save_path = "/Users/hr0283/Projects/StateSpaceAnalysis.jl/example";
end

push!(LOAD_PATH, pwd());
push!(LOAD_PATH, "$(pwd())/../");
push!(LOAD_PATH, root_path);
if run_cluster
    println(LOAD_PATH)
end
# =============================================================




# LOAD PACKAGES =============================================================
using StateSpaceAnalysis
using Accessors
using Random
using LinearAlgebra
using Dates
using Revise # this is for development
# =============================================================




# CONFIGURE SYSTEM =============================================================
BLAS.set_num_threads(1);   # single threaded blas still faster
set_zero_subnormals(true); # avoid subnormals (don't think this matters much)

rand_seed = 99; # set random seed
Random.seed!(rand_seed); 

println("\n========== SYSTEM INFO ==========")
try
    display(versioninfo(verbose=true))
catch
    try
        println("Julia Version = $(VERSION)")
        display(versioninfo())
    catch
    end
end
println("BLAS config = $(BLAS.get_config())")
println("BLAS threads = $(BLAS.get_num_threads())")
println("ARGS: $(ARGS)")
println("========================================\n")
# =============================================================




# SET PARAMETERS ==============================================================
S = core_struct(
    prm=param_struct(
        
        seed = rand_seed,
        model_name = "test",
        changelog = "run test",
        load_name = "example",
        load_path = "/Users/hr0283/Projects/StateSpaceAnalysis.jl/example/example-data",
        pt_list = 1:1, # always has to be range

        max_iter_em = run_cluster ? 2e4 : 500,
        test_iter = 100,
        early_stop = true,

        x_dim_fast = round.(Int64, 16:16:128),
        
        root_path = root_path,
        save_path = save_path,
        do_save = run_cluster ? true : false, 

        y_transform = "PCA",
        PCA_ratio = .99,

        do_trial_sel = true, # only epochs with current & previous accurate

        ssid_fit = length(ARGS) > 2 ? ARGS[3] : "fit", # fit, load
        ssid_save =  length(ARGS) > 3 ? parse(Bool, ARGS[4]) : false, # SAVE SSID AND THEN EXIT

        ssid_type = :CVA,
        ssid_lag = run_cluster ? 128 : 24,
        ), 

    dat=data_struct(

        sel_event = 2:4,

        pt = 1, # pt default
        x_dim = 24 , # x_dim default

        basis_name = "bspline",
        spline_gap = 5, # number of samples between spline knots
        norm_basis = false, # normalize every timepoint in basis with 2-norm

        pred_list = [
            "taskRepeat", "taskSwitch", 
            "switch",
            "RT", "prevRT", 
            "cueColor", "cueShape", "cueRepeat",
            ],

        pred0_list = [
            "prevTask", "RT", "prevRT", 
            ],


        ),

    res=results_struct(),

    est=estimates_struct(),

    mdl=model_struct(),

);
println("--- changelog: ",S.prm.changelog, " ---\n\n")
#  =======================================================================




# FIT THE MODEL =======================================================================
@reset S.res.startTime_all = Dates.format(now(), "mm/dd/yyyy HH:MM:SS");
println("Starting fit at $(S.res.startTime_all)")


# PREPROCESS ======================================
@reset S = StateSpaceAnalysis.preprocess_fit(S);

# @reset S = StateSpaceAnalysis.init_param_rand(S); # helfpul for debugging

# =================================================


# Subspace Identification (SSID) ==================
if S.prm.ssid_fit == "fit"

     # fit SSID
     @reset S = StateSpaceAnalysis.launch_SSID(S);

elseif S.prm.ssid_fit == "load"

    # load previously-fit SSID
    @reset S = StateSpaceAnalysis.load_SSID(S);

end
# ================================================


# Expectation Maximization (EM) ==================
@reset S = StateSpaceAnalysis.launch_EM(S);
# ================================================



@reset S.res.endTime_all = Dates.format(now(), "mm/dd/yyyy HH:MM:SS");
println("Finished fit at $(Dates.format(now(), "mm/dd/yyyy HH:MM:SS"))")
#  =======================================================================









# PLOT DIAGNOSTICS =======================================================================

do_plots = false
if do_plots
    try

        # plot loglik traces
        StateSpaceAnalysis.plot_loglik_traces(S)

        
        # plot posterior predictive checks        
        StateSpaceAnalysis.plot_avg_pred(S)


        # plot model
        StateSpaceAnalysis.plot_params(S)


    catch
    end
end
#  =======================================================================




# SAVE FIT =======================================================================
if S.prm.do_save

    println("\n========== SAVING FIT ==========")

    save_results(S);

else
    println("\n========== *NOT* SAVING FIT ==========")
end

println("=================================\n")

#  =======================================================================
