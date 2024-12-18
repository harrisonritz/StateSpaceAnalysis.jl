



function fit_EM(S); 
    """
        fit_EM(S::core_struct) -> core_struct

    Fit a linear-Gaussian state space model using the Expectation-Maximization (EM) algorithm.

    # Arguments
    - `S`: Core structure containing model parameters, data, and results

    # Description
    Implements the EM algorithm to estimate optimal parameters by iteratively:
    1. E-step: Computes expected sufficient statistics given current parameters 
    2. M-step: Updates parameters to maximize expected log-likelihood
    3. Evaluates convergence using training and/or test log-likelihood

    Key steps per iteration:
    - Calls `ESTEP!` to compute expectations
    - Calls `MSTEP` to update parameters
    - Computes log-likelihood and R² metrics
    - Checks convergence criteria

    The function tracks convergence by monitoring changes in log-likelihood and R² on held-out test data.
    Stops when likelihood improvements fall below `S.prm.train_threshold` or `S.prm.test_threshold`.

    # Returns
    Updated `core_struct` with:
    - Estimated model parameters in `S.mdl`
    - Fit metrics in `S.res`
    - Final expectations in `S.est`
    """

    # start the clock
    @reset S.res.startTime_em = Dates.format(now(), "mm/dd/yyyy HH:MM:SS");

    # check that estimates are initialized
    if all(S.est.xx_init .== 0) || all(S.est.yy_obs .== 0);
        @reset S.est = deepcopy(set_estimates(S));
    end


    # main EM loop ===================================================================
    for em_iter = 1:S.prm.max_iter_em

        # ==== E-STEP ================================================================
        @inline StateSpaceAnalysis.ESTEP!(S);

        # ==== M-STEP ================================================================
        @reset S.mdl = deepcopy(StateSpaceAnalysis.MSTEP(S));

        # ==== TOTAL LOGLIK ==========================================================
        StateSpaceAnalysis.total_loglik!(S)
        

        # check loglik ==========================================================

        # confirm loglik is increasing
        if (em_iter > 1)  && (S.res.total_loglik[em_iter] < S.res.total_loglik[em_iter-1])
            println("warning: total loglik decreased (Δll: $(round(S.res.total_loglik[end] - S.res.total_loglik[end-1],digits=3)))")
        end

        # test loglik every N iters
        if mod(em_iter, S.prm.test_iter) == 0

            @reset S.est = deepcopy(set_estimates(S));
            StateSpaceAnalysis.test_loglik!(S);
            push!(S.res.test_R2_proj, ll_R2(S, S.res.test_loglik[end], S.res.null_loglik[end]));    

            if length(S.res.test_loglik) > 1
                println("[$(em_iter)] total ll: $(round(S.res.total_loglik[em_iter],digits=2)) // test ll: $(round(S.res.test_loglik[end],digits=2)), Δll: $(round(S.res.total_loglik[end] - S.res.total_loglik[end-1],digits=2)) // test R2:$(round(S.res.test_R2_proj[end],digits=4))")
            else
                println("[$(em_iter)] total ll: $(round(S.res.total_loglik[em_iter],digits=2)) // test ll: $(round(S.res.test_loglik[end],digits=2)), test R2:$(round(S.res.test_R2_proj[end],digits=4))")
            end

        end


        # check for convergence

        # total loglik covergence
        if (em_iter > S.prm.check_train_iter) && 
            ((S.res.total_loglik[end] - S.res.total_loglik[end-1]) < S.prm.train_threshold)

            println("\n----- converged! -----")
            println("Δ total loglik: $(S.res.total_loglik[end] - S.res.total_loglik[end-1])")
            if length(S.res.test_loglik) > 1
                println("Δ test loglik: $(S.res.test_loglik[end] - S.res.test_loglik[end-1])")
            end
            println("\n\n")

            break

        end

        # test loglik covergence
        if (length(S.res.test_loglik) > 1) &&
            (S.prm.early_stop && ((S.res.test_loglik[end] - S.res.test_loglik[end-1]) < S.prm.test_threshold))

            println("\n----- converged! -----")
            println("Δ total loglik: $(S.res.total_loglik[end] - S.res.total_loglik[end-1])")
            println("Δ test loglik: $(S.res.test_loglik[end] - S.res.test_loglik[end-1])")
            println("\n\n")


            break

        end

        # garbage collect every 10 iter
        if (mod(em_iter,10) == 0) && Sys.islinux() 
            ccall(:malloc_trim, Cvoid, (Cint,), 0);
            ccall(:malloc_trim, Int32, (Int32,), 0);
            GC.gc(true);
        end


    end


    # final test fit ===========================================================
    @reset S.est = deepcopy(set_estimates(S));        
    StateSpaceAnalysis.test_loglik!(S);
    P = StateSpaceAnalysis.posterior_sse(S, S.dat.y_test, S.dat.y_test_orig, S.dat.u_test, S.dat.u0_test);

    push!(S.res.test_R2_proj, ll_R2(S, S.res.test_loglik[end], S.res.null_loglik[end]));    
    push!(S.res.test_R2_orig, 1.0 - (P.sse_orig[1] / S.res.null_sse_orig[end]));
    
    @reset S.res.fwd_R2_proj = 1.0 .- (P.sse_fwd_proj ./ S.res.null_sse_proj[1]);            
    @reset S.res.fwd_R2_orig = 1.0 .- (P.sse_fwd_orig ./ S.res.null_sse_orig[1]);

    push!(S.res.test_sse_proj, P.sse_proj[1]);    
    push!(S.res.test_sse_orig, P.sse_orig[1]);
    # ===========================================================

     
    

    println("[END] total ll: $(round(S.res.total_loglik[end],digits=2)) // test ll: $(round(S.res.test_loglik[end],digits=2)) // test R2: proj:$(round(S.res.test_R2_proj[end],digits=4)), orig:$(round(S.res.test_R2_orig[end],digits=4))")
    println("")


    @reset S.res.mdl_em = deepcopy(S.mdl);
    @reset S.res.endTime_em = Dates.format(now(), "mm/dd/yyyy HH:MM:SS");



    return S

end




# ===== E-STEP =================================================================

function ESTEP!(S)
    
    """
        ESTEP!(S::core_struct)

    Perform the E-step of the EM algorithm by computing expected sufficient statistics.

    # Arguments
    - `S`: Core structure containing current model parameters and data

    # Description
    Computes expectations in three phases:
    1. Calls `estimate_cov!` for state covariances
    2. Calls `init_moments!` to setup statistics
    3. Calls `estimate_mean!` for state means
    4. Updates moment matrices for M-step

    # Implementation Notes
    - Updates expectations in-place in `S.est`
    - Uses information vesion of Kalman filtering and RTS smoothing
    - Accumulates sufficient statistics across trials
    - Required for parameter updates in `MSTEP`

    Modifies `S.est` in-place with computed expectations.
    """

    # estimate latent covariance ==================
    @inline estimate_cov!(S);


    # initialize moments ==========================
    init_moments!(S);


    # estimate latent mean  ======================
    @inline estimate_mean!(S);   

end







# ===== ESTIMATE LATENT COVARIANCE =================================================================

function estimate_cov!(S)
    """
        estimate_cov!(S::core_struct)

    Estimate state covariances via Kalman filtering and RTS smoothing.

    # Arguments
    - `S`: Core structure containing model parameters and data

    # Description
    For each trial:
    1. Initializes with prior covariance P0
    2. Calls `filter_cov!` for forward pass 
    3. Calls `smooth_cov!` for backward pass
    4. Accumulates cross-time covariances

    # Implementation Notes
    - Updates `S.est.pred_cov`, `S.est.filt_cov`, `S.est.smooth_cov`
    - Since covaraince filter-smoother does not depend on inputs or observations, 
        it is computed once for all trials and then scaled by the number of trials.
    - Uses PDMat type to ensure positive definiteness
    """

    # filter cov ================================
    S.est.pred_cov[1] = deepcopy(S.mdl.P0);
    S.est.pred_icov[1] = deepcopy(S.mdl.iP0);
    S.est.filt_cov[1] = inv(S.mdl.CiRC + S.mdl.iP0); 

    @inline filter_cov!(S);


    # smooth cov  ===============================
    S.est.smooth_xcov .= zeros(S.dat.x_dim, S.dat.x_dim);
    S.est.smooth_cov[end] = S.est.filt_cov[end];

    @inline smooth_cov!(S);

end


function filter_cov!(S)
    """
        filter_cov!(S::core_struct)

    Forward pass covariance estimation using information form.

    # Arguments
    - `S`: Core structure containing model parameters

    # Description
    For each timepoint:
    1. Predicts next covariance: P(t+1|t) = APA' + Q
    2. Computes information matrices
    3. Updates using information filter equations

    # Implementation Notes
    - Updates `S.est.pred_cov`, `S.est.pred_icov`, `S.est.filt_cov`
    - Alternative implementation: `filter_cov_KF!` for standard filter (no reccomended)
    - Uses PDMat type for numerical stability
    """


    # filter covariance ================================
    @inbounds @views for tt in eachindex(S.est.filt_cov)[2:end]

        S.est.pred_cov[tt] = PDMat(X_A_Xt(S.est.filt_cov[tt-1], S.mdl.A) + S.mdl.Q);
        S.est.pred_icov[tt] = inv(S.est.pred_cov[tt]);
        S.est.filt_cov[tt] = inv(S.mdl.CiRC + S.est.pred_icov[tt]);

    end
   
end


function filter_cov_KF!(S)
    """
        smooth_cov!(S::core_struct)

    Backward pass covariance estimation using RTS smoother.

    # Arguments
    - `S`: Core structure containing filtered estimates

    # Description
    For each timepoint (backwards):
    1. Computes smoothing gain matrix G(t)
    2. Updates covariance: P(t|1:T) = P(t|t) + G[P(t+1|1:T) - P(t+1|t)]G'
    3. Accumulates cross-covariance for dynamics

    # Implementation Notes
    - Updates `S.est.smooth_cov`, `S.est.smooth_xcov`
    - Requires valid filtered covariances from `filter_cov!`
    - Uses temporary storage in `S.est.xdim2_temp`
    """


    # filter covariance ================================
    @inbounds @views for tt in eachindex(S.est.filt_cov)[2:end]

        S.est.pred_cov[tt] = PDMat(X_A_Xt(S.est.filt_cov[tt-1], S.mdl.A) + S.mdl.Q);
        S.est.pred_icov[tt] = inv(S.est.pred_cov[tt]);

        S.est.K[:,:,tt] = S.est.pred_cov[tt]*S.mdl.C' / 
                                    tol_PD(X_A_Xt(S.est.pred_cov[tt], S.mdl.C) + S.mdl.R);


        S.est.filt_cov[tt] =  tol_PD(X_A_Xt(S.est.pred_cov[tt], I - S.est.K[:,:,tt]*S.mdl.C) .+ 
                                        X_A_Xt(S.mdl.R, S.est.K[:,:,tt]));

    end
   
end



@inline function smooth_cov!(S)
    """
        smooth_cov!(S::core_struct)

    Compute backward pass covariances using RTS smoother.

    # Arguments
    - `S`: Core structure containing filtered estimates

    # Description
    For each timepoint (backwards):
    1. Computes smoother gain matrix
    2. Updates smoothed state covariance
    3. Accumulates cross-time covariance

    Updates in-place:
    - `S.est.smooth_cov`: Smoothed state covariances
    - `S.est.smooth_xcov`: Cross-time state covariances

    Uses efficient matrix operations with temporary storage in `S.est.xdim2_temp`.
    """



    # smooth covariance ================================
    @inbounds @views for tt in eachindex(S.est.filt_cov)[end-1:-1:1]

        # reverse kalman gain
        mul!(S.est.G[:,:,tt], S.est.filt_cov[tt], S.mdl.A', 1.0, 0.0);
        S.est.G[:,:,tt] /= S.est.pred_cov[tt+1];

        # smoothed covariancess
        mul!(S.est.xdim2_temp, S.est.G[:,:,tt], S.mdl.A, 1.0, 0.0);
        S.est.smooth_cov[tt] = PDMat(X_A_Xt(S.est.smooth_cov[tt+1] + S.mdl.Q, S.est.G[:,:,tt]) .+ 
                                     X_A_Xt(S.est.filt_cov[tt], I - S.est.xdim2_temp));

        # smoothed cross-cov
        mul!(S.est.smooth_xcov, S.est.G[:,:,tt], S.est.smooth_cov[tt+1], 1.0, 1.0);

    end

end






# ===== ESTIMATE LATENT MEAN =================================================================

function estimate_mean!(S)
    """
        estimate_mean!(S::core_struct)

    Estimate latent state means via Kalman filtering and RTS smoothing.

    # Arguments
    - `S`: Core structure containing model parameters and data

    # Description
    For each trial:
    1. Initializes predicted means using initial inputs
    2. Calls `filter_mean!` for forward pass
    3. Calls `smooth_mean!` for backward pass
    4. Accumulates results across trials

    # Implementation Notes
    - Updates `S.est.pred_mean`, `S.est.filt_mean`, `S.est.smooth_mean`
    - Works in conjunction with covariances from `estimate_cov!`
    - Uses efficient matrix operations with preallocation
    """

    @inbounds @views for tl in axes(S.dat.y_train,3)   

        # Initial condition
        mul!(S.est.pred_mean[:,1], S.mdl.B0, S.dat.u0_train[:,tl], 1.0, 0.0);


        # transform data ================================
        S.est.u_cur .= S.dat.u_train[:,1:end-1,tl];
        S.est.u0_cur .= S.dat.u0_train[:,tl];
        mul!(S.est.Bu, S.mdl.B, S.dat.u_train[:,:,tl], 1.0, 0.0);
        mul!(S.est.CiRY, S.mdl.CiR, S.dat.y_train[:,:,tl], 1.0, 0.0);
        S.est.y_cur .= S.dat.y_train[:,:,tl];


        # filter mean ===================================
        mul!(S.est.xdim_temp, S.mdl.iP0, S.est.pred_mean[:,1], 1.0, 0.0);
        S.est.xdim_temp .+= S.est.CiRY[:,1];
        mul!(S.est.filt_mean[:,1], S.est.filt_cov[1], S.est.xdim_temp, 1.0, 0.0);

        @inline filter_mean!(S);
    

        # smooth mean  ==================================
        S.est.smooth_mean[:,end] .= S.est.filt_mean[:, end];

        @inline smooth_mean!(S);


        # estimate moments ==============================
        mul!(S.est.xy_obs, S.est.smooth_mean, S.dat.y_train[:,:,tl]', 1.0, 1.0);

        estimate_moments!(S);

    end

    # format moments
    S.est.xx_dyn_PD[1] = tol_PD(S.est.xx_dyn);
    S.est.xx_obs_PD[1] = tol_PD(S.est.xx_obs);

end



@inline function filter_mean!(S)
    """
        filter_mean!(S::core_struct)

    Forward pass state estimation using information form Kalman filter.

    # Arguments
    - `S`: Core structure containing model parameters and data

    # Description
    For each timepoint:
    1. Predicts next state: x̂(t+1|t) = Ax̂(t|t) + Bu(t)
    2. Updates with observations using information form
    3. Uses covariances from `filter_cov!`

    # Implementation Notes
    - Updates `S.est.pred_mean`, `S.est.filt_mean` 
    - Requires valid `S.est.pred_cov`, `S.est.filt_cov`
    - Alternative implementation: `filter_mean_KF!` for standard form (not reccomended)
    """


    # filter mean [slow]
    @inbounds @views for tt in eachindex(S.est.pred_icov)[2:end]

        mul!(S.est.pred_mean[:,tt], S.mdl.A, S.est.filt_mean[:,tt-1], 1.0, 0.0);
        S.est.pred_mean[:,tt] .+= S.est.Bu[:,tt-1];

        mul!(S.est.xdim_temp, S.est.pred_icov[tt], S.est.pred_mean[:,tt], 1.0, 0.0);
        S.est.xdim_temp .+= S.est.CiRY[:,tt];

        mul!(S.est.filt_mean[:,tt], S.est.filt_cov[tt], S.est.xdim_temp, 1.0, 0.0);

    end


end





function filter_mean_KF!(S)

    # filter mean [slow]
    @inbounds @views for tt in eachindex(S.est.pred_icov)[2:end]

        mul!(S.est.pred_mean[:,tt], S.mdl.A, S.est.filt_mean[:,tt-1], 1.0, 0.0);
        S.est.pred_mean[:,tt] .+= S.est.Bu[:,tt-1];

        S.est.y_cur[:,tt] .-= S.mdl.C*S.est.pred_mean[:,tt]
        mul!(S.est.filt_mean[:,tt], S.est.K[:,:,tt], S.est.y_cur[:,tt], 1.0, 0.0);
        S.est.filt_mean[:,tt] .+= S.est.pred_mean[:,tt];

    end


end




@inline function smooth_mean!(S)
    """
        smooth_mean!(S::core_struct)

    Backward pass state estimation using RTS smoother.

    # Arguments
    - `S`: Core structure containing filtered estimates

    # Description
    For each timepoint (backwards):
    1. Computes smoothing gain from `smooth_cov!`
    2. Updates state: x̂(t|1:T) = x̂(t|t) + G(t)[x̂(t+1|1:T) - Ax̂(t|t) - Bu(t)]
    3. Uses filtered means from `filter_mean!`

    # Implementation Notes
    - Updates `S.est.smooth_mean`
    - Requires valid `S.est.filt_mean`, `S.est.smooth_cov`
    - Uses smoothing matrices from `smooth_cov!`
    """

    # smooth mean
    @inbounds @views for tt in eachindex(S.est.pred_icov)[end-1:-1:1]

        S.est.xdim_temp .= S.est.smooth_mean[:,tt+1] .- S.est.pred_mean[:,tt+1];
        @inline mul!(S.est.smooth_mean[:,tt], S.est.G[:,:,tt], S.est.xdim_temp, 1.0, 0.0);
        S.est.smooth_mean[:,tt] .+= S.est.filt_mean[:,tt];

    end


end





# ===== ESTIMATE MODEL MOMENTS =================================================================

function init_moments!(S)
    """
        init_moments!(S::core_struct)

    Initialize sufficient statistics matrices needed for parameter estimation in the M-step.

    # Arguments
    - `S`: Core structure containing smoothed state estimates

    # Description
    Initializes three sets of moment matrices:

    Initial State Moments:
    - `xy_init`: Cross moments between initial inputs (u₀) and states (x₁)
    - `yy_init`: Initial state covariance scaled by number of trials
    - `n_init`: Number of initial state observations

    Dynamics Moments:
    - `xx_dyn`: Augmented state-input covariance [x;u][x;u]'
    - `xy_dyn`: Cross moments between current and next states
    - `yy_dyn`: Next state covariance 
    - `n_dyn`: Number of state transitions

    Observation Moments:
    - `xx_obs`: State covariance across all timepoints
    - `xy_obs`: Cross moments between states and observations
    - `n_obs`: Total number of observations

    # Implementation Notes
    - All matrices are initialized with appropriate dimensions
    - Covariance matrices are scaled by number of trials
    - Uses efficient in-place operations with `.=`
    - Leverages precomputed smoothed state estimates
    """


    # init ===============================================
    S.est.xy_init .= zeros(S.dat.u0_dim, S.dat.x_dim);
    S.est.yy_init .= S.est.smooth_cov[1] .* S.dat.n_train;
    S.est.n_init .= copy(S.dat.n_train);


    # dyn ===============================================
    S.est.xx_dyn .= zeros(S.dat.x_dim + S.dat.u_dim, S.dat.x_dim + S.dat.u_dim);
    S.est.xx_dyn[1:S.dat.x_dim,1:S.dat.x_dim] .= sum(S.est.smooth_cov[1:end-1]) .* S.dat.n_train;
    S.est.xx_dyn[(S.dat.x_dim+1):end, (S.dat.x_dim+1):end] .= copy(S.est.uu_dyn);
    
    S.est.xy_dyn .= zeros(S.dat.x_dim + S.dat.u_dim, S.dat.x_dim);
    S.est.xy_dyn[1:S.dat.x_dim,:] .= S.est.smooth_xcov*S.dat.n_train;

    S.est.yy_dyn .= sum(S.est.smooth_cov[2:end]) * S.dat.n_train;

    S.est.n_dyn .= (S.dat.n_steps-1) * S.dat.n_train;


    # obs ===============================================
    S.est.xx_obs .= sum(S.est.smooth_cov) * S.dat.n_train;
    S.est.xy_obs .= zeros(S.dat.x_dim, S.dat.y_dim);
    S.est.n_obs .= S.dat.n_steps * S.dat.n_train;


end


@views function estimate_moments!(S)
    """
        estimate_moments!(S::core_struct)

    Compute sufficient statistics for parameter updates in the M-step.

    # Arguments
    - `S`: Core structure containing smoothed state estimates

    # Description
    Accumulates moments needed for parameter estimation:

    Initial state moments:
    - `S.est.xy_init`: Cross-covariance between initial inputs and states
    - `S.est.yy_init`: Covariance of initial states

    Dynamic moments:
    - `S.est.xx_dyn`: Augmented state-input covariance [x;u][x;u]'
    - `S.est.xy_dyn`: Cross-covariance between current and next states
    - `S.est.yy_dyn`: Covariance of next states 
    - `S.est.n_dyn`: Number of transition pairs

    Observation moments:
    - `S.est.xx_obs`: State covariance
    - `S.est.xy_obs`: Cross-covariance between states and observations
    - `S.est.n_obs`: Number of observations

    # Implementation Note
    Uses efficient matrix operations with preallocation for speed.
    Updates all moment matrices in-place within `S.est`.
    """
    
    
    # convienence variables =======================
    S.est.x_cur .= S.est.smooth_mean[:,1:end-1];
    S.est.x_next .= S.est.smooth_mean[:,2:end];


    # # initials moments =======================
    mul!(S.est.xy_init, S.est.u0_cur, S.est.x_cur[:,1]', 1.0, 1.0);
    mul!(S.est.yy_init, S.est.x_cur[:,1], S.est.x_cur[:,1]', 1.0, 1.0);


    # # dynamics moments =======================
    # # x_dyn * x_dyn
    mul!(S.est.xx_dyn[1:S.dat.x_dim,1:S.dat.x_dim], S.est.x_cur, S.est.x_cur', 1.0, 1.0);
    mul!(S.est.xx_dyn[1:S.dat.x_dim,(S.dat.x_dim+1):end], S.est.x_cur, S.est.u_cur', 1.0, 1.0);
    mul!(S.est.xx_dyn[(S.dat.x_dim+1):end, 1:S.dat.x_dim], S.est.u_cur, S.est.x_cur', 1.0, 1.0);

    # # x_dyn * y_dyn
    mul!(S.est.xy_dyn[1:S.dat.x_dim,:], S.est.x_cur, S.est.x_next', 1.0, 1.0);
    mul!(S.est.xy_dyn[S.dat.x_dim+1:end,:], S.est.u_cur, S.est.x_next', 1.0, 1.0);

    # # y_dyn * y_dyn
    mul!(S.est.yy_dyn, S.est.x_next, S.est.x_next', 1.0, 1.0);


    # # emissions moments =======================
    mul!(S.est.xx_obs, S.est.smooth_mean, S.est.smooth_mean', 1.0, 1.0);


end






# ===== M-STEP =================================================================

function MSTEP(S)::model_struct
    """
        MSTEP(S::core_struct) -> model_struct 

    Perform the M-step of the EM algorithm by maximizing expected log-likelihood.

    # Arguments
    - `S`: Core structure containing expectations from E-step

    # Description
    Updates model parameters in closed form:
    1. Initial state distribution (B0, P0)
    2. State dynamics (A, B, Q)
    3. Observation model (C, R)

    Uses sufficient statistics:
    - Initial moments (xy_init, yy_init)
    - Dynamics moments (xx_dyn, xy_dyn, yy_dyn)
    - Observation moments (xx_obs, xy_obs)

    Each parameter update includes regularization controlled by:
    - Prior strength parameters (`S.prm.lam_*`): B0, A, B, C
    - Degrees of freedom parameters (`S.prm.df_*`): (P0, Q, R)
    - Structure constraints (`S.prm.*_type`): (P0, Q, R)
    By default, priors are not used for the noise covariances.

    # Implementation Notes
    - Returns new model_struct with updated parameters
    - Applies regularization via prior parameters
    - Ensures valid covariance formats
    - Uses efficient matrix operations

    # Returns
    New `model_struct` containing updated parameters
    """
    
    # initials ===============================================
    # Mean
    W = ((S.est.xx_init + S.prm.lam_B0) \ S.est.xy_init)';
    B0 = W[:, 1:S.dat.u0_dim]

    # Covariance
    Wxy = W*S.est.xy_init;
    P0e = (S.est.yy_init .- Wxy .- Wxy' .+ X_A_Xt(S.est.xx_init, W) .+ W*S.prm.lam_B0*W' + (S.prm.df_P0 * S.prm.mu_P0)) / 
            ((S.est.n_init[1] + S.prm.df_P0) - size(S.est.xx_init,1));


    P0 = format_noise(P0e, S.prm.P0_type);

    


    # latents ===============================================
    # Mean
    W = ((S.est.xx_dyn_PD[1] + S.prm.lam_AB) \ S.est.xy_dyn)';
    A = W[:, 1:S.dat.x_dim];
    B = W[:, (S.dat.x_dim+1):end];

    # Covariance
    Wxy = W*S.est.xy_dyn;
    Qe = (S.est.yy_dyn .- Wxy .- Wxy' .+ X_A_Xt(S.est.xx_dyn_PD[1], W) .+ W*S.prm.lam_AB*W' + (S.prm.df_Q * S.prm.mu_Q)) / 
        ((S.est.n_dyn[1] + S.prm.df_Q) - size(S.est.xx_dyn,1));

    Q = format_noise(Qe, S.prm.Q_type);




    # emissions ===============================================
    # Mean
    W = ((S.est.xx_obs_PD[1] + S.prm.lam_C) \ S.est.xy_obs)';
    C = deepcopy(W);

    # Covariance
    Wxy = W*S.est.xy_obs;
    Re = (S.est.yy_obs .- Wxy .- Wxy' .+ X_A_Xt(S.est.xx_obs_PD[1], W) .+ W*S.prm.lam_C*W' + (S.prm.df_R * S.prm.mu_R)) / 
            ((S.est.n_obs[1] + S.prm.df_R) - size(S.est.xx_obs,1));

    R = format_noise(Re, S.prm.R_type);



    # reconstruct model
    mdl = set_model(
        A = A,
        B = B,
        Q = Q,
        C = C,
        R = R,
        B0 = B0,
        P0 = P0,
        );


    return mdl

end






