# plotting function
```
NOTE: not implmented yet (need better way to import Plots)
```

# using Plots

function generate_PPC(S,trial)


    # get posterior estimates
    P = StateSpaceAnalysis.posterior_all(   S, 
                                    S.dat.y_test[:,:,trial:trial],
                                    S.dat.y_test_orig[:,:,trial:trial],  
                                    S.dat.u_test[:,:,trial:trial], 
                                    S.dat.u0_test[:,trial:trial],
                                    );

    # get mean trajectory
    mean_xhat, mean_yhat  = StateSpaceAnalysis.generate_dlds_trials(S.mdl.A, S.mdl.B, S.mdl.Q,
                                            S.mdl.C, S.mdl.R, 
                                            S.mdl.B0, S.mdl.P0,
                                            S.dat.u_test[:,:,trial], S.dat.u0_test[:,trial],
                                            S.dat.n_steps, 1);                
    mean_orig_yhat = remix(S, mean_yhat[:,:,1]);


    return P, mean_xhat, mean_yhat, mean_orig_yhat


end


function plot_trial_pred(S, trial)


    P, _, mean_yhat, mean_orig_yhat = generate_PPC(S,trial);


    println("pred-obs R2 (projected) = $(round(cor(vec(P.pred_proj_y), vec(P.obs_proj_y)).^2, digits=4))")
    println("pred-obs R2 (original) = $(round(cor(vec(P.pred_orig_y), vec(P.obs_orig_y)).^2, digits=4))")

    println("mean-obs R2 (projected) = $(round(cor(vec(mean_yhat), vec(P.obs_proj_y)).^2, digits=4))")
    println("mean-obs R2 (original) = $(round(cor(vec(mean_orig_yhat), vec(P.obs_orig_y)).^2, digits=4))")

    # plot data
    stdY = std(P.obs_orig_y)/2;
    global plt = plot(label="", title="channel predictions", xlabel="time", ylabel="voltage (stacked electrodes)", yticks=false);
    for cc = 1:4:size(P.obs_orig_y,1)
        global plt = plot!(P.obs_orig_y[cc,:,1] .+ cc*stdY, label="", linewidth = 3, color = :black)
        global plt = plot!(P.pred_orig_y[cc,:,1] .+ cc*stdY, label="", linestyle = :dash, linewidth = 2, color = :red)
        # global plt = plot!(mean_orig_yhat[cc,:] .+ cc*stdY, label="", linewidth = 2, color = :magenta, opacity=0.66)
    end

    plot(plt, size = (800,800))


end




function plot_avg_pred(S)


    # get posterior estimates
    P = StateSpaceAnalysis.posterior_mean(   S, 
                                    S.dat.y_test,
                                    S.dat.y_test_orig,  
                                    S.dat.u_test, 
                                    S.dat.u0_test,
                                    );


    mean_obs_orig_y = mean(S.dat.y_test_orig, dims=3)[:,:,1]
    mean_pred_orig_y = remix(S,mean(cat([S.mdl.C*P.pred_mean[:,:,ii] for ii in axes(P.pred_mean,3)]..., dims=3),dims=3)[:,:,1])

    plt = plot(mean_obs_orig_y[1:4:end,:]', label="", title="Mean Observed Original Y", xlabel="time", ylabel="voltage", yticks=[], color=:black, linewidth=2)
    plt = plot!(mean_pred_orig_y[1:4:end,:]', label="", title="Mean Observed Original Y", xlabel="time", ylabel="voltage", yticks=[], color=:red, linestyle=:dash, linewidth=2)
    plot(plt,  size=(1200,800))

end





function plot_temp_cov(S, trial)


    # get posterior estimates
    P = StateSpaceAnalysis.posterior_mean(   S, 
                                    S.dat.y_test[:,:,trial:trial],
                                    S.dat.y_test_orig[:,:,trial:trial],  
                                    S.dat.u_test[:,:,trial:trial], 
                                    S.dat.u0_test[:,trial:trial],
                                    );

   
    res_x = (S.mdl.A*P.smooth_mean[:,1:end-1,1] .+ S.mdl.B*S.dat.u_test[:,1:end-1,trial]) - P.smooth_mean[:,2:end,1];      
    res_y = (S.mdl.C*P.smooth_mean[:,:,1]) - S.dat.y_test[:,:,trial];
    
    plt_x = heatmap((res_x'*res_x)/(S.dat.n_steps-1), title="temporal covariance of residuals (x)", xlabel="time", ylabel="time", color=:viridis)
    plt_y = heatmap((res_y'*res_y)/(S.dat.n_steps), title="temporal covariance of residuals (y)", xlabel="time", ylabel="time", color=:viridis)
    plot(plt_x, plt_y, layout=(1,2), size=(800,400))

end







function plot_loglik_traces(S)


        # plt_refine = plot(S.res.refine_total_loglik, legend=false, title="refine total loglik", xlabel="EM Iteration", ylabel="loglik")


        if isempty(S.res.total_loglik)

            plot(plt_refine, size=(800,800))

        else

            plt_em=plot(S.res.total_loglik, legend=false, title="EM total loglik", xlabel="EM Iteration", ylabel="loglik")
            
            plt_em_sep=plot(zscore(S.res.init_loglik), label="init", title="EM total loglik (seperate)", xlabel="EM Iteration", ylabel="loglik")
            plt_em_sep=plot!(zscore(S.res.dyn_loglik), label="dyn")
            plt_em_sep=plot!(zscore(S.res.obs_loglik), label="obs")

            plt_test=plot(S.res.test_loglik, legend=false, title="test loglik", xlabel="Iteration", ylabel="loglik")

            plt_R2_proj=plot(S.res.test_R2_proj, legend=false, title="test R2 (proj)", xlabel="Iteration", ylabel="R2")
            # plt_R2_orig=plot(S.res.test_R2_orig, legend=false, title="test R2 (orig)", xlabel="Iteration", ylabel="R2")

            plt_R2_fwd=plot(S.res.fwd_R2_proj, label="proj", title="forward pred R2", xlabel="lookahead", ylabel="R2", ylims=(0.0,1.0)) 
            plt_R2_fwd=plot!(S.res.fwd_R2_orig, label="orig", title="forward pred R2", xlabel="lookahead", ylabel="R2") 


            plot(plt_em,plt_em_sep, plt_test, plt_R2_proj, plt_R2_fwd, layout=(3,2), size=(1200,1200))

        end


end





function plot_model(S; save=false)


    plt_refine = plot(S.res.refine_total_loglik, legend=false, title="refine total loglik", xlabel="EM Iteration", ylabel="loglik")


    if isempty(S.res.total_loglik)

        plot(plt_refine, size=(800,800))

    else

        plt_em=plot(S.res.total_loglik, legend=false, title="EM total loglik", xlabel="EM Iteration", ylabel="loglik")
        plt_test=plot(S.res.test_loglik, legend=false, title="EM test loglik", xlabel="EM Iteration", ylabel="loglik")
        plt_R2=plot(S.res.test_R2, legend=false, title="EM test R2", xlabel="EM Iteration", ylabel="R2")

        plot(plt_refine, plt_em, plt_test, plt_R2,  layout=(2,2), size=(800,800))

    end


end

function plot_params(S)


    sym_col(x) = (-1, 1).*maximum(abs, x)

    plot_square(x, title) = heatmap(x, title=title, color=:coolwarm, aspect_ratio=1, clims=sym_col(x))
    plot_rect(x,title) = heatmap(x, title=title, color=:coolwarm, clims=sym_col(x))
    
    plt_A = plot_square(S.mdl.A, "A");

    plot_Ai = plot(title="eig(A)", aspect_ratio=1);
    plt_Ai = scatter!(eigen(S.mdl.A).values, label="", marker=:circle, color=:proj, markersize=5, legend=false);
    plot_Ai = plot!(sin.(-pi:.001:pi), cos.(-pi:.001:pi), color=:black, label="")

    plt_B = plot_rect(S.mdl.B, "B");
    plt_Bc = heatmap(   LowerTriangular(cor(S.mdl.B)), title="cor(B)", color=:coolwarm, aspect_ratio=1, clims=(-1,1), 
                        xticks=(1:length(S.dat.pred_name),S.dat.pred_name), xrotation = 90, 
                        yticks=(1:length(S.dat.pred_name),S.dat.pred_name))

    ul = reshape(S.dat.u_train, S.dat.u_dim, S.dat.n_steps*S.dat.n_train)';
    plt_Uc = heatmap(   LowerTriangular(cov(ul)), title="cor(U)", color=:coolwarm, aspect_ratio=1, clims=(-.33,.33), 
                        xticks=(1:length(S.dat.pred_name),S.dat.pred_name), xrotation = 90, 
                        yticks=(1:length(S.dat.pred_name),S.dat.pred_name));


    plt_Q = plot_square(S.mdl.Q, "Q");

    plt_C = plot_rect(S.mdl.C, "C");
    plt_CiR = plot_rect(S.mdl.CiR, "CiR");
    plt_CiRC = plot_square(S.mdl.CiRC, "CiRC");
    plt_R = plot_square(S.mdl.R, "R");

    plt_B0 = plot_rect(S.mdl.B0, "B0");
    plt_P0 = plot_square(S.mdl.P0, "P0");

    plot(   plt_A, plt_Ai, plt_Q,  
            plt_B, plt_Bc, plt_Uc, 
            plt_C, plt_CiR, plt_CiRC, 
            plt_R, plt_B0, plt_P0, 
            layout=(4,4), size=(3000, 3000))
    
end