# plotting function
using Plots

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


    println("pred-obs R2 (whitened) = $(round(cor(vec(P.pred_white_y), vec(P.obs_white_y)).^2, digits=4))")
    println("pred-obs R2 (original) = $(round(cor(vec(P.pred_orig_y), vec(P.obs_orig_y)).^2, digits=4))")

    println("mean-obs R2 (whitened) = $(round(cor(vec(mean_yhat), vec(P.obs_white_y)).^2, digits=4))")
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

            plt_R2_white=plot(S.res.test_R2_white, legend=false, title="test R2 (white)", xlabel="Iteration", ylabel="R2")
            # plt_R2_orig=plot(S.res.test_R2_orig, legend=false, title="test R2 (orig)", xlabel="Iteration", ylabel="R2")

            plt_R2_fwd=plot(S.res.fwd_R2_white, label="white", title="forward pred R2", xlabel="lookahead", ylabel="R2", ylims=(0.0,1.0)) 
            plt_R2_fwd=plot!(S.res.fwd_R2_orig, label="orig", title="forward pred R2", xlabel="lookahead", ylabel="R2") 


            plot(plt_em,plt_em_sep, plt_test, plt_R2_white, plt_R2_fwd, layout=(3,2), size=(1200,1200))

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






function report_params(S)


    println("\n========== A ========== ")
    display(S.mdl.A)
    println("\n========== B ========== ")
    display(S.mdl.B)
    println("\n========== Q ========== ")
    display(S.mdl.Q)

    println("\n========== C ========== ")
    display(S.mdl.C)
    println("\n========== R ========== ")
    display(S.mdl.R)

    println("\n========== B0 ========== ")
    display(S.mdl.B0)
    println("\n========== P0 ========== ")
    display(S.mdl.P0)

end




function plot_params(S)


    sym_col(x) = (-1, 1).*maximum(abs, x)

    plot_square(x, title) = heatmap(x, title=title, color=:coolwarm, aspect_ratio=1, clims=sym_col(x))
    plot_rect(x,title) = heatmap(x, title=title, color=:coolwarm, clims=sym_col(x))
    
    plt_A = plot_square(S.mdl.A, "A");

    plot_Ai = plot(title="eig(A)", aspect_ratio=1);
    plt_Ai = scatter!(eigen(S.mdl.A).values, label="", marker=:circle, color=:white, markersize=5, legend=false);
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




function plot_bal_params(S)


    sym_col(x) = (-1, 1).*maximum(abs, x)

    plot_square(x, title) = heatmap(x, title=title, color=:coolwarm, aspect_ratio=1, clims=sym_col(x))
    plot_rect(x,title) = heatmap(x, title=title, color=:coolwarm, clims=sym_col(x))
    

    sys = ss(S.mdl.A, S.mdl.B, S.mdl.C, 0, S.dat.dt)
    _,G,T = balreal(sys);


    plt_A = plot_square(T*S.mdl.A/T, "A");

    plot_Ai = plot(title="eig(A)", aspect_ratio=1);
    plt_Ai = scatter!(eigen(T*S.mdl.A/T).values, label="", marker=:circle, color=:white, markersize=5, legend=false);
    plot_Ai = plot!(sin.(-pi:.001:pi), cos.(-pi:.001:pi), color=:black, label="")

    plt_B = plot_rect(T*S.mdl.B, "B");
    plt_Bc = heatmap(   LowerTriangular(cor(T*S.mdl.B)), title="cor(B)", color=:coolwarm, aspect_ratio=1, clims=(-1,1), 
                        xticks=(1:length(S.dat.pred_name),S.dat.pred_name), xrotation = 90, 
                        yticks=(1:length(S.dat.pred_name),S.dat.pred_name));

    ul = reshape(S.dat.u_train, S.dat.u_dim, S.dat.n_steps*S.dat.n_train)';
    plt_Uc = heatmap(   LowerTriangular(cov(ul)), title="cor(U)", color=:coolwarm, aspect_ratio=1, clims=(-.33,.33), 
                        xticks=(1:length(S.dat.pred_name),S.dat.pred_name), xrotation = 90, 
                        yticks=(1:length(S.dat.pred_name),S.dat.pred_name));


    plt_Q = plot_square(Matrix(X_A_Xt(S.mdl.Q, T)), "Q");

    plt_C = plot_rect(S.mdl.C/T, "C");
    plt_CiR = plot_rect(T\S.mdl.CiR, "CiR");
    plt_CiRC = plot_square(Matrix(X_A_Xt(S.mdl.CiRC,T)), "CiRC");
    plt_R = plot_square(S.mdl.R, "R");

    plt_B0 = plot_rect(T*S.mdl.B0, "B0");
    plt_P0 = plot_square(Matrix(X_A_Xt(S.mdl.P0,T)), "P0");

    plt_G = plot(G, title="G", label="",  color=:black, linewidth=3);
    hline!(plt_G, [0], color=:red, linestyle=:dash, label="", linewidth=3)


    plot(plt_A, plt_Ai, plt_B,plt_Bc, plt_Q, plt_C, plt_CiR, plt_CiRC, plt_R, plt_B0, plt_P0,plt_G, layout=(3,4), size=(2000, 1500))
    
end



function plot_input_diffusion(S; input_name = "task", mod_name = "task@switch", norm_type="state")


    cond_sel = findall(S.dat.pred_name .== input_name);
    mod_sel = findall(S.dat.pred_name .== mod_name);
    base_sel = S.dat.n_misc + 1:length(cond_sel);

    A = S.mdl.A;
    B = S.mdl.B;
    C = S.mdl.C;

    O = dlyap(A', C'*C);
    # O = S.mdl.CiRC;
    # O = C'*C;
    N = dlyap(A, S.mdl.Q);
    # N = S.mdl.Q;



    # inputs
    b_0 = B[:,cond_sel];
    if isempty(mod_sel)
        b_modP = zeros(S.dat.x_dim,length(cond_sel));
        b_modN = zeros(S.dat.x_dim,length(cond_sel));
    else
        b_modP = b_0 + B[:,mod_sel];
        b_modN = b_0 - B[:,mod_sel];
    end
    # b_modP = b_0 + B[:,mod_sel];
    # b_modN = b_0 - B[:,mod_sel];

    basis = S.dat.u_train[base_sel,:,1];

    BB_0, BB_P, BB_N = zeros(size(A)), zeros(size(A)), zeros(size(A));
    W_0, W_P, W_N = zeros(S.dat.n_steps,1), zeros(S.dat.n_steps,1), zeros(S.dat.n_steps,1);
    
    for cc in eachindex(W_0)

        basis_tt = Diagonal(basis[:,cc]);


        BB_0 = A*BB_0*A' + b_0*basis_tt*b_0';
        BB_P = A*BB_P*A' + b_modP*basis_tt*b_modP';
        BB_N = A*BB_N*A' + b_modN*basis_tt*b_modN';

        if norm_type == "state"
            W_0[cc] = tr(N\BB_0);
            W_P[cc] = tr(N\BB_P);
            W_N[cc] = tr(N\BB_N);
        elseif norm_type == "obs"
            W_0[cc] = tr(O*BB_0);
            W_P[cc] = tr(O*BB_P);
            W_N[cc] = tr(O*BB_N);
        elseif norm_type == "none"
            W_0[cc] = tr(BB_0);
            W_P[cc] = tr(BB_P);
            W_N[cc] = tr(BB_N);
        end

    end


    global plt_in=plot(0:S.dat.n_steps,[0;W_0], title="input diffusion ($(input_name))", xlabel="time", ylabel="trace(W_t)", legend=false, color=:black, linewidth=3)
    if S.dat.epoch[1] == 1
        plt_in=vline!([findfirst(S.dat.epoch .== 2)], color=:red, linestyle=:dash, label="")
    end
    plt_in=vline!([findlast(S.dat.epoch .== 2)], color=:red, linestyle=:dash)
    plt_in=vline!([findlast(S.dat.epoch .== 3)], color=:red, linestyle=:dash)

    global plt_mod=plot(0:S.dat.n_steps,[0;W_P], title="input contrast diffusion ($(input_name))", xlabel="time", ylabel="trace(W_t)", label="$(mod_name) +", color=:green, linewidth=3)
    plt_mod=plot!(0:S.dat.n_steps,[0;W_N],label="$(mod_name) -", color=:red, linewidth=3)
    
    if S.dat.epoch[1] == 1
        plt_mod=vline!([findfirst(S.dat.epoch .== 2)], color=:red, linestyle=:dash, label="")
    end
    plt_mod=vline!([findlast(S.dat.epoch .== 2)], color=:red, linestyle=:dash, label="")
    plt_mod=vline!([findlast(S.dat.epoch .== 3)], color=:red, linestyle=:dash, label="")

    plot(plt_in, plt_mod, size=(1800,600))


end





function plot_2input_diffusion(S; input1_name = "task", input2_name = "task@switch", norm_type="state")


    cond1_sel = findall(S.dat.pred_name .== input1_name);
    cond2_sel = findall(S.dat.pred_name .== input2_name);
    base_sel = S.dat.n_misc + 1:length(cond1_sel);

    A = S.mdl.A;
    B = S.mdl.B;
    C = S.mdl.C;

    O = dlyap(A', C'*C);
    N = dlyap(A, S.mdl.Q);



    # inputs
    b_1 = B[:,cond1_sel];
    b_2 = B[:,cond2_sel];
    basis = S.dat.u_train[base_sel,:,1];

    BB_1, BB_2 = zeros(size(A)), zeros(size(A));
    W_1, W_2 = zeros(S.dat.n_steps,1), zeros(S.dat.n_steps,1);
    
    for cc in eachindex(W_1)

        basis_tt = Diagonal(basis[:,cc]);

        BB_1 = A*BB_1*A' + b_1*basis_tt*b_1';
        BB_2 = A*BB_2*A' + b_2*basis_tt*b_2';

        if norm_type == "state"
            W_1[cc] = tr(N\BB_1);
            W_2[cc] = tr(N\BB_2);
        elseif norm_type == "obs"
            W_1[cc] = tr(O*BB_1);
            W_2[cc] = tr(O*BB_2);
        elseif norm_type == "none"
            W_1[cc] = tr(BB_1);
            W_2[cc] = tr(BB_2);
        end

    end


    global plt_in=plot(0:S.dat.n_steps,[0;W_1], title="input diffusion", xlabel="time", ylabel="trace(W_t)", label="$(input1_name)", color=:green, linewidth=3)

    plt_in=plot!(0:S.dat.n_steps,[0;W_2], label="$(input2_name)", color=:red, linewidth=3)
    if S.dat.epoch[1] == 1
        plt_in=vline!([findfirst(S.dat.epoch .== 2)], color=:red, linestyle=:dash, label="")
    end
    plt_in=vline!([findlast(S.dat.epoch .== 2)], color=:red, linestyle=:dash, label="")

    if S.dat.epoch[end] == 4
        plt_in=vline!([findlast(S.dat.epoch .== 3)], color=:red, linestyle=:dash, label="")
    end

    plot(plt_in, size=(800,400))


end