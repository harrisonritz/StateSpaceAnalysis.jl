



function fit_EM(S); 
    """
        fit_EM(S)

    Fit a probabilistic state-space model using the Expectation-Maximization (EM) algorithm.

    # Arguments
    - `S`: structure containing the parameters, data, and results of the model fit.

    # Returns
    - `S`: structure containing the parameters, data, and results of the model fit.
   
    # Description

    The `fit_EM` function implements the EM algorithm to estimate the parameters of a probabilistic model. 
    It iteratively performs:
        1. Expectation (E) step, computing the expected value of the log-likelihood with respect to the current parameter estimates
        2. Maximization (M) step, updating the parameters to maximize this expected log-likelihood
        3. Checks for convergence of the log-likelihood and stops if the change is below a threshold.

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
            push!(S.res.test_R2_white, ll_R2(S, S.res.test_loglik[end], S.res.null_loglik[end]));    

            if length(S.res.test_loglik) > 1
                println("[$(em_iter)] total ll: $(round(S.res.total_loglik[em_iter],digits=2)) // test ll: $(round(S.res.test_loglik[end],digits=2)), Δll: $(round(S.res.total_loglik[end] - S.res.total_loglik[end-1],digits=2)) // test R2:$(round(S.res.test_R2_white[end],digits=4))")
            else
                println("[$(em_iter)] total ll: $(round(S.res.total_loglik[em_iter],digits=2)) // test ll: $(round(S.res.test_loglik[end],digits=2)), test R2:$(round(S.res.test_R2_white[end],digits=4))")
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

    push!(S.res.test_R2_white, ll_R2(S, S.res.test_loglik[end], S.res.null_loglik[end]));    
    push!(S.res.test_R2_orig, 1.0 - (P.sse_orig[1] / S.res.null_sse_orig[end]));
    
    @reset S.res.fwd_R2_white = 1.0 .- (P.sse_fwd_white ./ S.res.null_sse_white[1]);            
    @reset S.res.fwd_R2_orig = 1.0 .- (P.sse_fwd_orig ./ S.res.null_sse_orig[1]);

    push!(S.res.test_sse_white, P.sse_white[1]);    
    push!(S.res.test_sse_orig, P.sse_orig[1]);
    # ===========================================================

     
    

    println("[END] total ll: $(round(S.res.total_loglik[end],digits=2)) // test ll: $(round(S.res.test_loglik[end],digits=2)) // test R2: white:$(round(S.res.test_R2_white[end],digits=4)), orig:$(round(S.res.test_R2_orig[end],digits=4))")
    println("")


    @reset S.res.mdl_em = deepcopy(S.mdl);
    @reset S.res.endTime_em = Dates.format(now(), "mm/dd/yyyy HH:MM:SS");



    return S

end




# ===== E-STEP =================================================================

function ESTEP!(S)
    
    """
    run E-step for individual participants

    """

    # estimate latent covariance ==================
    @inline estimate_cov!(S);


    # initialize moments ==========================
    init_moments!(S);


    # estimate latent mean  ======================
    # init
    @inline estimate_mean!(S);   

end







# ===== ESTIMATE LATENT COVARIANCE =================================================================

function estimate_cov!(S)

    # filter cov ================================
    S.est.pred_cov[1] = deepcopy(S.mdl.P0);
    S.est.pred_icov[1] = deepcopy(S.mdl.iP0);
    S.est.filt_cov[1] = inv(S.mdl.CiRC + S.mdl.iP0); 

    filter_cov!(S);


    # smooth cov  ===============================
    S.est.smooth_xcov .= zeros(S.dat.x_dim, S.dat.x_dim);
    S.est.smooth_cov[end] = S.est.filt_cov[end];

    smooth_cov!(S);

end


function filter_cov!(S)

    # filter covariance ================================
    @inbounds @views for tt in eachindex(S.est.filt_cov)[2:end]

        S.est.pred_cov[tt] = PDMat(X_A_Xt(S.est.filt_cov[tt-1], S.mdl.A) + S.mdl.Q);
        S.est.pred_icov[tt] = inv(S.est.pred_cov[tt]);
        S.est.filt_cov[tt] = inv(S.mdl.CiRC + S.est.pred_icov[tt]);

    end
   
end


function filter_cov_KF!(S)
    # standard KF

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

        filter_mean!(S);
    

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

    # smooth mean
    @inbounds @views for tt in eachindex(S.est.pred_icov)[end-1:-1:1]

        S.est.xdim_temp .= S.est.smooth_mean[:,tt+1] .- S.est.pred_mean[:,tt+1];
        @inline mul!(S.est.smooth_mean[:,tt], S.est.G[:,:,tt], S.est.xdim_temp, 1.0, 0.0);
        S.est.smooth_mean[:,tt] .+= S.est.filt_mean[:,tt];

    end


end





# ===== ESTIMATE MODEL MOMENTS =================================================================

function init_moments!(S)


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






