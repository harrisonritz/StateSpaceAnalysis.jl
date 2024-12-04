
# ===== LOGLIK =================================================================

ll_R2(S, test_loglik, null_loglik) = 1.0 - exp((2.0 /(S.dat.n_test*S.dat.n_steps*S.dat.y_dim)) * (null_loglik - test_loglik));



log_post_v0(n,v,v0,vN,lam0,lamN,Sig0,SigN) =    -0.5*n*v*log(2pi) .+
                                                0.5*v*logdet(lam0) .+ 
                                                -0.5*v*logdet(lamN) .+
                                                0.5*v0*logdet(0.5 .* Sig0) .+
                                                -0.5*vN*logdet(0.5 .* SigN) .+
                                                SpecialFunctions.loggamma(0.5 .* v0) .+ 
                                                -SpecialFunctions.loggamma(0.5 .* vN);


log_post(n,v,vN,lam0,lamN,SigN) =   -0.5*n*v*log(2pi) .+
                                    0.5*v*logdet(lam0) .+ 
                                    -0.5*v*logdet(lamN) .+
                                    -0.5*vN*logdet(0.5 .* SigN) .+
                                    -SpecialFunctions.loggamma(0.5 .* vN);







function init_lik(S)

    n = S.est.n_init[1];
    v = S.dat.x_dim;
    p = S.dat.u0_dim;

    v0 = S.prm.df_P0;
    vN = v0 + (n - p);

    lam0 = S.prm.lam_B0(p);
    lamN = lam0 + S.est.xx_init;

    Sig0 = Matrix(S.prm.mu_P0(v) * v0);
    SigN = S.mdl.P0 * vN;

    if v0 > 0
        log_p = log_post_v0(n,v,v0,vN,lam0,lamN,Sig0,SigN) 
    else
        log_p = log_post(n,v,vN,lam0,lamN,SigN) 
    end

    return log_p

end



function dyn_lik(S)

    n = S.est.n_dyn[1];
    v = S.dat.x_dim;
    p = S.dat.x_dim + S.dat.u_dim;

    v0 = S.prm.df_Q;
    vN = v0 + (n - p);

    lam0 = Matrix(S.prm.lam_AB(p));
    lamN = lam0 + S.est.xx_dyn;

    Sig0 = Matrix(S.prm.mu_Q(v) * v0);
    SigN = S.mdl.Q * vN;
    
    if v0 > 0
        log_p = log_post_v0(n,v,v0,vN,lam0,lamN,Sig0,SigN) 
    else
        log_p = log_post(n,v,vN,lam0,lamN,SigN) 
    end

    return log_p

end



function obs_lik(S)

    n = S.est.n_obs[1];
    v = S.dat.y_dim;
    p = S.dat.x_dim;

    v0 = S.prm.df_R;
    vN = v0 + (n - p);

    lam0 = Matrix(S.prm.lam_C(p));
    lamN = lam0 + S.est.xx_obs;

    Sig0 = Matrix(S.prm.mu_R(v) * v0);
    SigN = S.mdl.R * vN;

    if v0 > 0
        log_p = log_post_v0(n,v,v0,vN,lam0,lamN,Sig0,SigN) 
    else
        log_p = log_post(n,v,vN,lam0,lamN,SigN) 
    end

    return log_p

end




                  


function total_loglik!(S)

    # advance total loglik
    push!(S.res.init_loglik, 0.0);
    push!(S.res.dyn_loglik, 0.0);
    push!(S.res.obs_loglik, 0.0);
    push!(S.res.total_loglik, 0.0);

    # get logliks
    S.res.init_loglik[end] = init_lik(S);
    S.res.dyn_loglik[end] = dyn_lik(S);
    S.res.obs_loglik[end] = obs_lik(S);

    # total loglik
    S.res.total_loglik[end] = S.res.init_loglik[end] .+ S.res.dyn_loglik[end] .+ S.res.obs_loglik[end];

end

function total_loglik(S)

    total_loglik = 0.0;
    total_loglik += init_lik(S)
    total_loglik += dyn_lik(S)
    total_loglik += obs_lik(S)

    return total_loglik

end





function test_loglik!(S);

    # advance test loglik 
    push!(S.res.test_loglik, 0.0);
    len_test_loglik = length(S.res.test_loglik);

    # == filter covariance ==
    filter_cov!(S);

    # get mean & loglik
    @inbounds @views for tl in axes(S.dat.y_test,3)

        # set X0
        mul!(S.est.pred_mean[:,1], S.mdl.B0, S.dat.u0_test[:,tl], true , false);

        # transform data
        mul!(S.est.Bu, S.mdl.B, S.dat.u_test[:,:,tl]);
        mul!(S.est.CiRY, S.mdl.CiR, S.dat.y_test[:,:,tl]);

        # filter mean =========
        @inline filter_mean!(S);

        # get loglik
        mul!(S.est.test_mu, S.mdl.C, S.est.pred_mean, 1.0, 0.0);
        @inbounds @views for tt in axes(S.est.filt_mean,2)

            S.est.test_sigma[1] = PDMat(X_A_Xt(S.est.pred_cov[tt], S.mdl.C) + S.mdl.R);
            S.res.test_loglik[len_test_loglik] += logpdf(MvNormal(S.est.test_mu[:,tt], S.est.test_sigma[1]), S.dat.y_test[:,tt,tl]);

        end

    end

end





function test_loglik(S);

    # init test loglik 
    test_loglik = 0.0;

    # == filter covariance ==
    filter_cov!(S);

    # get mean & loglik
    @inbounds @views for tl in axes(S.dat.y_test,3)

        # set X0
        mul!(S.est.pred_mean[:,1], S.mdl.B0, S.dat.u0_test[:,tl], true , false);

        # transform data
        mul!(S.est.Bu, S.mdl.B, S.dat.u_test[:,:,tl], true , false);
        mul!(S.est.CiRY, S.mdl.CiR, S.dat.y_test[:,:,tl], true , false);

        # filter mean =========
        @inline filter_mean!(S);

        # get loglik
        mul!(S.est.test_mu, S.mdl.C, S.est.pred_mean, 1.0, 0.0);

        @inbounds @views for tt in axes(S.est.filt_mean,2)

            S.est.test_sigma[1] = tol_PD(X_A_Xt(S.est.pred_cov[tt], S.mdl.C) + S.mdl.R);
            test_loglik += logpdf(MvNormal(S.est.test_mu[:,tt], S.est.test_sigma[1]), S.dat.y_test[:,tt,tl]);

        end

    end

    return test_loglik

end





function test_orig_loglik(S);

    # init test loglik 
    test_loglik = 0.0;

    # == filter covariance ==
    filter_cov!(S);

    # get mean & loglik
    @inbounds @views for tl in axes(S.dat.y_test,3)

        # set X0
        mul!(S.est.pred_mean[:,1], S.mdl.B0, S.dat.u0_test[:,tl], true , false);

        # transform data
        mul!(S.est.Bu, S.mdl.B, S.dat.u_test[:,:,tl], true , false);
        mul!(S.est.CiRY, S.mdl.CiR, S.dat.y_test[:,:,tl], true , false);

        # filter mean =========
        @inline filter_mean!(S);

        # get loglik
        mul!(S.est.test_mu, S.mdl.C, S.est.pred_mean, 1.0, 0.0);
        @inbounds @views for tt in axes(S.est.filt_mean,2)

            S.est.test_sigma[1] = PDMat(X_A_Xt(S.est.pred_cov[tt], S.mdl.C) + S.mdl.R);
            test_loglik += logpdf(MvNormal(StateSpaceAnalysis.remix(S, S.est.test_mu[:,tt]), X_A_Xt(S.est.test_sigma[1], S.dat.W)), S.dat.y_orig_test[:,tt,tl]);

        end

    end

    return test_loglik

end








function null_loglik!(S)

    """ Compute null log-likelihood
    args:
        y_train: training data
        y_test: test data
    return:
        cov_ll: predict from mean
        rel_ll: predict from time-resolved mean/cov
        diff_ll: predict using temporal difference
        ar1_ll: predict using AR1 regression
    """

    # long format y
    yl_train = reshape(permutedims(S.dat.y_train, (2,3,1)), S.dat.n_steps*S.dat.n_train, S.dat.y_dim); # convert to long format (steps x trials, channels)
    yl_test = reshape(permutedims(S.dat.y_test, (2,3,1)), S.dat.n_steps*S.dat.n_test, S.dat.y_dim); # convert to long format (steps x trials, channels)

    # average first timepoint
    yl1_train = mean(S.dat.y_train[:,1,:], dims=2)';
    yl1_test = mean(S.dat.y_train[:,1,:], dims=2)';

    # previous timepoints
    ylp_train = deepcopy(S.dat.y_train);
    ylp_train[:,2:end,:] .= ylp_train[:,1:end-1,:];
    ylp_train[:,1,:] .= yl1_train';
    ylp_train = reshape(permutedims(ylp_train, (2,3,1)), S.dat.n_steps*S.dat.n_train, S.dat.y_dim);

    ylp_test = deepcopy(S.dat.y_test);
    ylp_test[:,2:end,:] .= ylp_test[:,1:end-1,:];
    ylp_test[:,1,:] .= yl1_test';
    ylp_test = reshape(permutedims(ylp_test, (2,3,1)), S.dat.n_steps*S.dat.n_test, S.dat.y_dim);

    # previous inputs
    up_train = deepcopy(S.dat.u_train);
    up_train[:,2:end,:] .= up_train[:,1:end-1,:];
    up_train[:,1,:] .= 0.0;
    up_train = reshape(permutedims(up_train, (2,3,1)), S.dat.n_steps*S.dat.n_train, S.dat.u_dim);

    up_test = deepcopy(S.dat.u_test);
    up_test[:,2:end,:] .= up_test[:,1:end-1,:];
    up_test[:,1,:] .= 0.0;
    up_test = reshape(permutedims(up_test, (2,3,1)), S.dat.n_steps*S.dat.n_test, S.dat.u_dim);

    calc_b(x,y) = [x; 1e-3I(size(x,2))] \ [y; zeros(size(x,2), size(y,2))];
    calc_res(x,y,b) = y .- x*b;
    calc_cov(r,df) = (r'*r) ./ (size(r,1) - df);

    n_models = 4;
    pred_train = Vector{Array}(undef, n_models)
    pred_test = Vector{Array}(undef, n_models)

    # models
    pred_train[1] = ones(size(yl_train,1));
    pred_test[1] = ones(size(yl_test,1));

    pred_train[2] = [ones(size(yl_train,1)) ylp_train]; 
    pred_test[2] = [ones(size(yl_test,1)) ylp_test]; 

    pred_train[3] = [ones(size(yl_train,1)) up_train];
    pred_test[3] = [ones(size(yl_test,1)) up_test];

    pred_train[4] = [ones(size(yl_train,1)) ylp_train up_train];
    pred_test[4] = [ones(size(yl_test,1)) ylp_test up_test];


    # get test likelihood
    test_zeros = zeros(S.dat.y_dim);

    for mm = 1:n_models

        pred_res = calc_res(pred_test[mm], yl_test, calc_b(pred_train[mm], yl_train));
        pred_cov = tol_PD(calc_cov(pred_res, size(pred_train[mm],2)));

        null_ll = 0.0;
        for tt in axes(yl_test,1)
                null_ll += logpdf(MvNormal(pred_res[tt,:], pred_cov), test_zeros);
        end

        S.res.null_sse_white[mm] = sumsqr(pred_res);
        S.res.null_mse_white[mm] = sumsqr(pred_res)./length(pred_res);

        remix_res = StateSpaceAnalysis.remix(S,pred_res');
        S.res.null_sse_orig[mm] = sumsqr(remix_res);
        S.res.null_mse_orig[mm] = sumsqr(remix_res)./length(remix_res);

        S.res.null_loglik[mm] = null_ll;

    end

end
