# fit setup functions



# read in arguements from command line
function read_args(S, arg_in)
    """
    read_args()

    Reads and processes the command-line arguments provided to the program.
    """

    
    # load arguements from command line
    println("ARGS: $arg_in")

    if isempty(arg_in)

        arg_num = 0;
        do_fast = true;

    else

        arg_num = parse(Int64, arg_in[1]);
        do_fast = parse(Bool, arg_in[2]); 

    end

    # get conditions
    if do_fast
        arg_list = [collect(S.prm.pt_list), S.prm.x_dim_fast];
    else
        arg_list = [collect(S.prm.pt_list), S.prm.x_dim_slow];
    end


    # get conditions
    if arg_num > 0

        indices = []
        for entry in arg_list
            push!(indices, mod(arg_num-1, length(entry)) + 1)
            arg_num = div(arg_num-1, length(entry)) + 1
        end
        conds = [entry[index] for (entry, index) in zip(arg_list, indices)];

        # assign conditions
        @reset S = assign_arguements(S, conds)
        @reset S.prm.arg_num = parse(Int64, arg_in[1]);

    end

    if S.prm.ssid_save == 1
        @reset S.dat.x_dim = S.prm.ssid_lag;
    end
    if S.prm.ssid_lag == -1
        @reset S.prm.ssid_lag = S.dat.x_dim;
    end

    if do_fast
        println("FAST=[$(minimum(S.prm.x_dim_fast))-$(maximum(S.prm.x_dim_fast))], CONDITION=$(S.prm.arg_num)/$(prod(length.(arg_list))): pt=$(S.dat.pt), x_dim=$(S.dat.x_dim)")
    else
        println("SLOW=[$(minimum(S.prm.x_dim_slow))-$(maximum(S.prm.x_dim_slow))], CONDITION=$(S.prm.arg_num)/$(prod(length.(arg_list))): pt=$(S.dat.pt), x_dim=$(S.dat.x_dim)")
    end

   
    @reset S.prm.save_name = "$(S.prm.model_name)_Pt$(S.dat.pt)_xdim$(S.dat.x_dim)";
    @reset S.prm.do_fast = do_fast;

    return S

end



# setup the directories for saving the results
function setup_path(S)
    """
    setup_path(S::Structure)

    Sets up the directories for saving the results.
    """


    # make output folders
    mkpath("$(S.prm.save_path)/fit-results/SSID-jls/$(S.prm.model_name)")
    mkpath("$(S.prm.save_path)/fit-results/figures/$(S.prm.model_name)")
    mkpath("$(S.prm.save_path)/fit-results/EM-jls/$(S.prm.model_name)")
    mkpath("$(S.prm.save_path)/fit-results/EM-mat/$(S.prm.model_name)")
    mkpath("$(S.prm.save_path)/fit-results/PPC-mat/$(S.prm.model_name)_PPC")

end




function load_data(S)
    # load data from file

    # load data
    if length(S.prm.load_path)>0
        raw_data = matread("$(S.prm.load_path)/$(S.prm.load_name)/$(S.prm.load_name)_$(S.dat.pt).mat")
    else
        error("no data path provided")
    end

    # check for data
    @assert haskey(raw_data, "y") "no y data"

    # get basic info
    @reset S.dat.ts = vec(raw_data["ts"]);          # time vector
    @reset S.dat.dt = raw_data["dt"];               # time step
    @reset S.dat.events = vec(raw_data["epoch"]);   # epoch events
    @reset S.dat.n_chans = size(raw_data["y"],1);   # number of channels
    @reset S.dat.n_trials = size(raw_data["y"],3);  # number of trials (before trial selection)


    @reset S.dat.trial = raw_data["trial"];
    if haskey(raw_data, "chanLocs")
        @reset S.dat.chanLocs = raw_data["chanLocs"]
    else
        println("couldn't load chanLocs")
    end

    # select trials (custom trial sel + train/test split)
    @reset S = deepcopy(select_trials(S));

    # select time points
    if !isempty(S.dat.events)
        if !isempty(S.dat.sel_event)
            @reset S.dat.sel_steps = vec(any(in.(S.dat.events', S.dat.sel_event),dims=1));
        else
            @reset S.dat.sel_steps = trues(size(raw_data["y"],2));
        end
        @reset S.dat.n_steps = sum(S.dat.sel_steps);
        @reset S.dat.events = S.dat.events[S.dat.sel_steps];
    else
        @reset S.dat.sel_steps = trues(size(raw_data["y"],2));
        @reset S.dat.n_steps = sum(S.dat.sel_steps);
        @reset S.dat.events = ones(size(raw_data["y"],2));
    end


    # setup EEG data
    # train
    @reset S.dat.y_train_orig = raw_data["y"][:,S.dat.sel_steps,S.dat.sel_train];
    @reset S.dat.n_train = size(S.dat.y_train_orig,3);

    # test
    @reset S.dat.y_test_orig =raw_data["y"][:,S.dat.sel_steps,S.dat.sel_test];
    @reset S.dat.n_test = size(S.dat.y_test_orig,3);

    # setup predictors
    @reset S.dat.n_pred = count(!isempty, S.dat.pred_list);
    @reset S.dat.n_pred0 = count(!isempty, S.dat.pred0_list);
    @reset S.dat.u0_dim = 1 + S.dat.n_pred0;
 
    
    return S

end







function build_inputs(S)

    println("")

    for fold in ["train", "test"]


        if fold == "train"

            sel = deepcopy(S.dat.sel_train);
            n_trials = deepcopy(S.dat.n_train);

        elseif fold == "test"

            sel = deepcopy(S.dat.sel_test);
            n_trials = deepcopy(S.dat.n_test);

        end



        # build within-trial design matrix        
        u_list = map(split_list, S.dat.pred_list);
        pred_cond = zeros(S.dat.n_pred, n_trials);

        center(A,d=1) = A .- mean(A,dims=d)

        if !isempty(S.dat.pred_list[1])
            for pp in eachindex(u_list)

                z_cond = ones(sum(sel));
                for uu in eachindex(u_list[pp]) # this loop allows for interaction terms

                    u_cond = S.dat.trial[u_list[pp][uu]];
                    z_cond .*= scale_input(u_cond, sel)

                end

                pred_cond[pp,:] .= z_cond;

            end
        end
      

        # build basis set
        u, n_bases, u_dim, pred_basis = create_input_basis(S, n_trials) # create basis set (custom.jl)
        @reset S.dat.n_bases = n_bases;
        @reset S.dat.u_dim = u_dim;
    
        # convolve predictors with basis set
        for bb = 1:S.dat.n_bases
            for uu in axes(pred_cond,1)
                u[(S.dat.n_misc + S.dat.n_bases) + ((uu-1)*S.dat.n_bases)+bb,:,:] .= u[S.dat.n_misc+bb,:,:] .* pred_cond[uu,:]';
            end
        end

        
        # check collinearity
        ul = deepcopy(reshape(u, S.dat.u_dim, S.dat.n_steps*n_trials)');
        f=svd(ul);

        # build initial state list
        u0_list = map(split_list, S.dat.pred0_list);
        u0 = zeros(S.dat.u0_dim, n_trials);
        u0[1,:] .= 1.0; # constant term

        if !isempty(S.dat.pred0_list[1])
            for pp in eachindex(u0_list)

                z_cond = ones(sum(sel));
                for uu in eachindex(u0_list[pp])
                    
                    u0_cond = S.dat.trial[u0_list[pp][uu]];
                    z_cond .*= scale_input(u0_cond, sel)

                end

                u0[pp+1,:] .= z_cond;

            end
        end


        pred_misc = []; # not using misc predictors for now
        if fold == "train"

            @reset S.dat.u_train = u;
            @reset S.dat.n_train = n_trials;
            @reset S.dat.pred_collin_train = f.S ./ f.S[end];
            @reset S.dat.pred_name = [pred_misc; pred_basis; vec(repeat(S.dat.pred_list, inner=(S.dat.n_bases,1)))];
       
            @reset S.dat.u0_train = u0;
            @reset S.dat.pred0_name = ["bias"; S.dat.pred0_list];

            @reset S.dat.u_train_cor = cor(ul);


            println("========== train fold inputs ==========")


        elseif fold == "test"

            @reset S.dat.u_test = u;
            @reset S.dat.n_test = n_trials;
            @reset S.dat.pred_collin_test = f.S ./ f.S[end];

            @reset S.dat.u0_test = u0;

            println("========== test fold fold inputs ==========")

        end

        # print out collinearity stats
        println("predictor mean quartiles: $(round(median(mean(ul, dims=1)), sigdigits=4)) +/- $(round(iqr(mean(ul, dims=1))/2, sigdigits=4))")
        println("predictor var quartiles: $(round(median(var(ul, dims=1)), sigdigits=4)) +/- $(round(iqr(var(ul, dims=1))/2, sigdigits=4))");
        println("collinearity metric (best=1, threshold=30): $(round(f.S[1]/f.S[end], sigdigits=4))"); 
        println("========================================\n")

    end 



    return S

end





function whiten(S)
# orthogonalize data

    y_long = reshape(S.dat.y_train_orig, S.dat.n_chans, S.dat.n_steps*S.dat.n_train);
    @reset S = deepcopy(transform_observations(S, y_long));


    # transform train ==================================
    y_train = zeros(S.dat.y_dim, S.dat.n_steps, S.dat.n_train);
    for tt in axes(y_train,3)
        y_train[:,:,tt] = StateSpaceAnalysis.demix(S, S.dat.y_train_orig[:,:,tt]);
    end
    @reset S.dat.y_train = y_train;


    # transform test ==================================
    y_test = zeros(S.dat.y_dim, S.dat.n_steps, S.dat.n_test);
    for tt in axes(y_test,3)
        y_test[:,:,tt] = StateSpaceAnalysis.demix(S, S.dat.y_test_orig[:,:,tt]);
    end
    @reset S.dat.y_test = y_test;


    return S

end






function test_rep_ESTEP(S)
#  run ESTEP twice and compare results (should be zero)
   

    StateSpaceAnalysis.ESTEP!(S);
    M1 = deepcopy(S.est);

    StateSpaceAnalysis.ESTEP!(S);
    M2 = deepcopy(S.est);


    rep_norm = zeros(0);

    push!(rep_norm, norm(M1.xx_init .- M2.xx_init))
    push!(rep_norm, norm(M1.xy_init .- M2.xy_init))
    push!(rep_norm, norm(M1.yy_init .- M2.yy_init));

    push!(rep_norm, norm(M1.xx_dyn .- M2.xx_dyn));
    push!(rep_norm, norm(M1.xy_dyn .- M2.xy_dyn));
    push!(rep_norm, norm(M1.yy_dyn .- M2.yy_dyn));

    push!(rep_norm, norm(M1.xx_obs .- M2.xx_obs))
    push!(rep_norm, norm(M1.xy_obs .- M2.xy_obs))
    push!(rep_norm, norm(M1.yy_obs .- M2.yy_obs));


    return rep_norm

end



function generate_lds_parameters(S, Q_noise, R_noise, P0_noise)::NamedTuple

    A = randn(S.dat.x_dim, S.dat.x_dim) + 5I;
    s,u = eigen(A); # get eigenvectors and eigenvals
    s = s/maximum(abs.(s))*.95; # set largest eigenvalue to lie inside unit circle (enforcing stability)
    s[real.(s) .< 0] = -s[real.(s) .< 0]; #set real parts to be positive (encouraging smoothness)
    A_sim = real(u*(Diagonal(s)/u));  # reconstruct A from eigs and eigenvectors

    # diagonal Q
    # Q =  Matrix(sqrt(Q_noise) * I(S.dat.x_dim)); 
    Q = sqrt(Q_noise) .* randn(S.dat.x_dim, S.dat.x_dim); 
    Q_sim = tol_PD(Q'*Q; tol=.1);

    B_sim = 0.5*randn(S.dat.x_dim, S.dat.u_dim);
    C_sim = 0.5*randn(S.dat.n_chans, S.dat.x_dim);

    R =  sqrt(R_noise) .* randn(S.dat.n_chans, S.dat.n_chans); 
    R_sim = tol_PD(R'*R; tol=.1);

    B0_sim = randn(S.dat.x_dim, S.dat.u0_dim);

    P = sqrt(P0_noise) .* randn(S.dat.x_dim, S.dat.x_dim);
    P0_sim = tol_PD(P'*P; tol=.1);
    

    sim = (A = A_sim, B = B_sim, C = C_sim, Q = Q_sim, R = R_sim, B0 = B0_sim, P0 = P0_sim);
    return sim

end





