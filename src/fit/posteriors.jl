# GENERATE POSTERIORS  ========================================================




function posterior_all(S, y, y_orig, u, u0)
    # all posteriors


    # initialize output ==========================
    n_trials = size(y,3);

    P = post_all(

            pred_mean = zeros(S.dat.x_dim, S.dat.n_steps,n_trials),
            filt_mean = zeros(S.dat.x_dim, S.dat.n_steps,n_trials),
            smooth_mean = zeros(S.dat.x_dim, S.dat.n_steps,n_trials),

            pred_cov = [[init_PD(S.dat.x_dim) for _ in 1:S.dat.n_steps] for _ in 1:n_trials],
            filt_cov = [[init_PD(S.dat.x_dim) for _ in 1:S.dat.n_steps] for _ in 1:n_trials],
            smooth_cov = [[init_PD(S.dat.x_dim) for _ in 1:S.dat.n_steps] for _ in 1:n_trials],

            obs_proj_y = zeros(S.dat.y_dim, S.dat.n_steps,n_trials),
            pred_proj_y = zeros(S.dat.y_dim, S.dat.n_steps,n_trials),
            filt_proj_y = zeros(S.dat.y_dim, S.dat.n_steps,n_trials),
            smooth_proj_y = zeros(S.dat.y_dim, S.dat.n_steps,n_trials),

            obs_orig_y = zeros(S.dat.n_chans, S.dat.n_steps,n_trials),
            pred_orig_y = zeros(S.dat.n_chans, S.dat.n_steps,n_trials),
            filt_orig_y = zeros(S.dat.n_chans, S.dat.n_steps,n_trials),
            smooth_orig_y = zeros(S.dat.n_chans, S.dat.n_steps,n_trials),

            sse_proj = [0.0],
            sse_orig = [0.0],

        );


    # estimate cov ===================================
    estimate_cov!(S)



    # estimate mean ==================================
    # @inbounds @views for tl in axes(y,3)   
    @inbounds for tl in axes(y,3)   

        # Initial condition
        mul!(S.est.pred_mean[:,1], S.mdl.B0, u0[:,tl], 1.0, 0.0);


        # transform data ================================
        S.est.u_cur .= u[:,1:end-1,tl][:,:,1];
        S.est.u0_cur .= u0[:,tl][:,1];
        mul!(S.est.Bu, S.mdl.B, u[:,:,tl][:,:,1], 1.0, 0.0);
        mul!(S.est.CiRY, S.mdl.CiR, y[:,:,tl][:,:,1], 1.0, 0.0);


        # filter mean ===================================
        S.est.xdim_temp .= S.est.CiRY[:,1] .+ S.mdl.iP0*S.est.pred_mean[:,1];
        @views mul!(S.est.filt_mean[:,1], S.est.filt_cov[1], S.est.xdim_temp, 1.0, 0.0);

        @inline filter_mean!(S);
    

        # smooth mean  ==================================
        S.est.smooth_mean[:,end] .= S.est.filt_mean[:, end];

        @inline smooth_mean!(S);




        # save results ==================================
        P.pred_mean[:,:,tl] .= S.est.pred_mean;
        P.pred_cov[tl] .= S.est.pred_cov;

        P.filt_mean[:,:,tl] .= S.est.filt_mean;
        P.filt_cov[tl] .= S.est.filt_cov;

        P.smooth_mean[:,:,tl] .= S.est.smooth_mean;
        P.smooth_cov[tl] .= S.est.smooth_cov;

        P.obs_proj_y[:,:,tl] .= y[:,:,tl];
        P.pred_proj_y[:,:,tl] .= S.mdl.C * S.est.pred_mean;
        P.filt_proj_y[:,:,tl] .= S.mdl.C * S.est.filt_mean;
        P.smooth_proj_y[:,:,tl] .= S.mdl.C * S.est.smooth_mean;
        
        if S.dat.W == zeros(0,0)
            P.obs_orig_y[:,:,tl] .= P.obs_proj_y[tl];
            P.pred_orig_y[:,:,tl] .= P.pred_proj_y[tl];
            P.filt_orig_y[:,:,tl] .= P.filt_proj_y[tl];
            P.smooth_orig_y[:,:,tl] .= P.smooth_proj_y[tl];
        else
            P.obs_orig_y[:,:,tl] .= y_orig[:,:,tl];
            P.pred_orig_y[:,:,tl] .= StateSpaceAnalysis.remix(S, S.mdl.C * S.est.pred_mean);
            P.filt_orig_y[:,:,tl] .= StateSpaceAnalysis.remix(S, S.mdl.C * S.est.filt_mean);
            P.smooth_orig_y[:,:,tl] .= StateSpaceAnalysis.remix(S, S.mdl.C * S.est.smooth_mean);
        end

        P.sse_proj[1] += sumsqr(P.obs_proj_y[tl] .-  P.pred_proj_y[tl]);
        P.sse_orig[1] += sumsqr(P.obs_orig_y[tl].-  P.pred_orig_y[tl]);

        for ii in 0:25

            pred_y = S.mdl.C * S.mdl.A^(ii) * S.est.pred_mean[:, 1:end-ii];

            if ii >0
                for bb in 1:ii
                    pred_y .+= S.mdl.C * S.mdl.A^(bb-1) * S.mdl.B * S.dat.u_train[:, (ii-bb+1):end-bb, tl];
                end
            end

            P.sse_fwd_proj[ii+1] += sumsqr(pred_y .-  P.obs_proj_y[:, (ii+1):end,tl]);
            P.sse_fwd_orig[ii+1] += sumsqr(StateSpaceAnalysis.remix(S, pred_y) .-  P.obs_orig_y[:, (ii+1):end,tl]);

        end

    end

    return P

end


function posterior_mean(S, y, y_orig, u, u0)
    # just posterior means

    # initialize output ==========================
    P = post_mean(
            pred_mean = zeros(S.dat.x_dim, S.dat.n_steps, size(y,3)),
            filt_mean = zeros(S.dat.x_dim, S.dat.n_steps, size(y,3)),
            smooth_mean = zeros(S.dat.x_dim, S.dat.n_steps, size(y,3)),
        );


    # estimate cov ===================================
    estimate_cov!(S)


    # estimate mean ==================================
    # @inbounds @views for tl in axes(y,3)   
    @inbounds @views for tl in axes(y,3)   

        # Initial condition
        mul!(S.est.pred_mean[:,1], S.mdl.B0, u0[:,tl], 1.0, 0.0);


        # transform data ================================
        S.est.u_cur .= u[:,1:end-1,tl][:,:,1];
        S.est.u0_cur .= u0[:,tl][:,1];
        mul!(S.est.Bu, S.mdl.B, u[:,:,tl][:,:,1], 1.0, 0.0);
        mul!(S.est.CiRY, S.mdl.CiR, y[:,:,tl][:,:,1], 1.0, 0.0);


        # filter mean ===================================
        S.est.xdim_temp .= S.est.CiRY[:,1] .+ S.mdl.iP0*S.est.pred_mean[:,1];
        mul!(S.est.filt_mean[:,1], S.est.filt_cov[1], S.est.xdim_temp, 1.0, 0.0);

        @inline filter_mean!(S);
    

        # smooth mean  ==================================
        S.est.smooth_mean[:,end] .= S.est.filt_mean[:, end];

        @inline smooth_mean!(S);


        # save results ==================================
        P.pred_mean[:,:,tl] .= S.est.pred_mean;
        P.filt_mean[:,:,tl] .= S.est.filt_mean;
        P.smooth_mean[:,:,tl] .= S.est.smooth_mean;

    end

    return P

end




function posterior_sse(S, y, y_orig, u, u0)
    # just posteriors SSE

    # initialize output ==========================
    n_trials = size(y,3);

    P = post_sse(
            sse_proj = [0.0],
            sse_orig = [0.0],
            sse_fwd_proj = zeros(26),
            sse_fwd_orig = zeros(26),
        );


    # estimate cov ===================================
    estimate_cov!(S)



    # estimate mean ==================================
    # @inbounds @views for tl in axes(y,3)   
    @inbounds @views for tl in axes(y,3)   

        # Initial condition
        mul!(S.est.pred_mean[:,1], S.mdl.B0, u0[:,tl], 1.0, 0.0);


        # transform data ================================
        S.est.u_cur .= u[:,1:end-1,tl][:,:,1];
        S.est.u0_cur .= u0[:,tl][:,1];
        mul!(S.est.Bu, S.mdl.B, u[:,:,tl][:,:,1], 1.0, 0.0);
        mul!(S.est.CiRY, S.mdl.CiR, y[:,:,tl][:,:,1], 1.0, 0.0);


        # filter mean ===================================
        S.est.xdim_temp .= S.est.CiRY[:,1] .+ S.mdl.iP0*S.est.pred_mean[:,1];
        mul!(S.est.filt_mean[:,1], S.est.filt_cov[1], S.est.xdim_temp, 1.0, 0.0);

        @inline filter_mean!(S);
    

        # smooth mean  ==================================
        S.est.smooth_mean[:,end] .= S.est.filt_mean[:, end];

        @inline smooth_mean!(S);



        # SAVE RESULTS ==================================
        P.sse_proj[1] += sumsqr(y[:,:,tl] .- S.mdl.C*S.est.pred_mean);
        P.sse_orig[1] += sumsqr(y_orig[:,:,tl] .-  StateSpaceAnalysis.remix(S, S.mdl.C * S.est.pred_mean));

        for ii in 0:25

            pred_y = S.mdl.C * S.mdl.A^(ii) * S.est.pred_mean[:, 1:end-ii];

            if ii >0
                for bb in 1:ii
                    pred_y .+= S.mdl.C * S.mdl.A^(bb-1) * S.mdl.B * S.dat.u_train[:, (ii-bb+1):end-bb, tl];
                end
            end

            P.sse_fwd_proj[ii+1] += sumsqr(pred_y .-  y[:,(ii+1):end,tl]);
            P.sse_fwd_orig[ii+1] += sumsqr(StateSpaceAnalysis.remix(S, pred_y) .-  y_orig[:,(ii+1):end,tl]);

        end

    end

    return P

end


