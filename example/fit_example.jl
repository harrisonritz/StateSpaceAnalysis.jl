
# LOAD PACKAGES =============================================================
using StateSpaceAnalysis
using Accessors
using Random
using LinearAlgebra
using Dates
using Revise # this is for development
# =============================================================



# SET PATHS =============================================================
run_cluster = false; # set cluster conditional here
if run_cluster # set cluster paths here
    save_path = "YOUR_CLUSTER_SAVE_PATH"; 
    load_path = "YOUR_CLUSTER_LOAD_PATH";
else
    save_path =  pkgdir(StateSpaceAnalysis, "example")
    load_path =  pkgdir(StateSpaceAnalysis, "example", "example-data")
end

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

        save_path = save_path,
        load_path = load_path,

        seed = rand_seed,
        model_name = "test",
        changelog = "run test",
        load_name = "example",
        pt_list = 1:1, # always has to be range

        max_iter_em = run_cluster ? 2e4 : 500,
        test_iter = 100,
        early_stop = true,

        x_dim_fast = round.(Int64, 16:16:128),
        
        do_save = run_cluster ? true : true, 

        y_transform = "PCA",
        PCA_ratio = .99,

        do_trial_sel = true, # only epochs with current & previous accurate

        ssid_fit = length(ARGS) > 2 ? ARGS[3] : "fit", # "fit"/"load"; "fit" new SSID, or "load" existing SSID
        ssid_save =  length(ARGS) > 3 ? parse(Bool, ARGS[4]) : false, # SAVE SSID AND THEN EXIT

        ssid_type = :CVA,
        ssid_lag = run_cluster ? 128 : 16,
        ), 

    dat=data_struct(

        sel_event = 2:4,

        pt = 1, # pt default
        x_dim = 16 , # x_dim default

        basis_name = "bspline",
        spline_gap = 5, # spline knots optimized for every n timesteps

        pred_list = [
            "task",
            ],

        pred0_list = [
            "prevTask",
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

# TODO: implement plotting functions
# do_plots = false
# if do_plots
#     try

#         # plot loglik traces
#         StateSpaceAnalysis.plot_loglik_traces(S)

        
#         # plot posterior predictive checks        
#         StateSpaceAnalysis.plot_avg_pred(S)


#         # plot model
#         StateSpaceAnalysis.plot_params(S)


#     catch
#     end
# end

#  =======================================================================




# SAVE FIT =======================================================================
if S.prm.do_save

    println("\n========== SAVING FIT ==========")

    StateSpaceAnalysis.save_results(S);

else
    println("\n========== *NOT* SAVING FIT ==========")
end

println("=================================\n")

#  =======================================================================
