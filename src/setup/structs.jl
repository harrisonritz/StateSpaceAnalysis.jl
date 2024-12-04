# structures



@kwdef struct param_struct

    # model name & changelog
    model_name::String = "test"
    save_name::String = "test"
    changelog::String = ""
    arg_num = 0

    # filename
    load_name::String = ""
    load_path::String = ""

    seed::Int64 = 99

    max_iter_em::Int64 = 2e4        # max iterations for EM
    test_iter::Int64 = 100          # compute test loglik every n iterations
    check_train_iter::Int64 = 100   # check total loglik after n iterations
    train_threshold::Float64 = 1    # total loglik stopping criterion
    early_stop::Bool = true         # stop early if test loglik doesn't improve
    test_threshold::Float64 = 1e-3  # test loglik stopping criterion

    pt_list::Union{UnitRange{Int64}, Vector{Int64}, Int64} = 1:1

    x_dim_fast::Array{Int64,1} = round.(Int64, 16:16:128)
    x_dim_slow::Array{Int64,1} = round.(Int64, 144:16:256)
    do_fast = true

    ssid_fit::String = "fit"
    ssid_save::Bool = false
    ssid_lag::Int64 = 10
    ssid_type::Symbol = :CVA

    y_transform::String = "PCA"
    PCA_ratio::Float64 = 0.99
    PCA_maxdim::Int64 = 1000

    do_trial_sel::Bool = false # select trials
    


    # priors
    lam_AB::UniformScaling{Float64} = 1e-6I
    lam_C::UniformScaling{Float64}= 1e-6I
    lam_B0::UniformScaling{Float64} = 1e-6I

    df_Q::Float64 = 0.0
    df_R::Float64 = 0.0
    df_P0::Float64 = 0.0

    mu_Q::UniformScaling{Float64} = .01I    # only matters if df_Q > 0
    mu_R::UniformScaling{Float64} = .001I   # only matters if df_R > 0
    mu_P0::UniformScaling{Float64} = .1I    # only matters if df_P0 > 0

    Q_refine_type::String = "full"
    R_refine_type::String = "full"
    P0_refine_type::String = "full"

    Q_type::String = "full"
    R_type::String = "full"
    P0_type::String = "full"

    save_path::String = pkgdir(StateSpaceAnalysis)
    do_save::Bool = false
    write_mat::Bool = true

end



@kwdef struct data_struct

    pt::Int64 = 1;

    x_dim::Int64 = 10;
    y_dim::Int64 = 0;
    u_dim::Int64 = 0;
    u0_dim::Int64 = 0;
    ts::Vector{Float64} = zeros(0);
    dt::Float64 = 0.0;
    events::Vector{Float64} = zeros(0);

    n_chans::Int64 = 0;
    n_steps::Int64 = 0;
    n_trials::Int64 = 0;
    chanLocs::Dict = Dict();

    trial::Dict = Dict();

    n_train::Int64 = 0;
    y_train_orig::Array{Float64,3} = zeros(n_chans, n_steps, n_train);
    y_train::Array{Float64,3} = zeros(y_dim, n_steps, n_train);
    u_train::Array{Float64,3} = zeros(u_dim, n_steps, n_train);
    u0_train::Matrix{Float64} = zeros(u0_dim, n_train);
    u_train_cor::Matrix{Float64} = zeros(0, 0);
    u_ssid::Array{Float64,3} = zeros(u_dim, n_steps, n_train);

    n_test::Int64 = 0;
    y_test_orig::Array{Float64,3} = zeros(n_chans, n_steps, n_test);
    y_test::Array{Float64,3} = zeros(y_dim, n_steps, n_test);
    u_test::Array{Float64,3} = zeros(u_dim, n_steps, n_test);
    u0_test::Matrix{Float64} = zeros(u0_dim, n_test);

    # select data
    sel_trial::BitArray{1} = falses(0);
    sel_train::BitArray{1} = falses(0);
    sel_test::BitArray{1} = falses(0);
    sel_steps::BitArray{1} = falses(0);
    sel_event::Vector{Int64} = zeros(0);


    # predictors
    pred_list::Vector{String} = [""];
    pred_name::Vector{String} = [""];
    n_pred::Int64 = 0;

    pred0_list::Vector{String} = [""];
    pred0_name::Vector{String} = [""];
    n_pred0::Int64 = 0;

    pred_collin_train::Array{Float64,1} = zeros(0);
    pred_collin_test::Array{Float64,1} = zeros(0);

    basis_name::String = "bspline";
    n_bases::Int64 = 1;
    n_splines::Int64 = 0;
    spline_gap::Float64 = 5;
    bin_skip::Int64 = 0;
    bin_width::Float64 = 0.050;
    n_misc::Int64 = 0;
    norm_basis::Bool = false;

    # transform y
    W::Matrix{Float64} = zeros(0,0);
    mu::Vector{Float64} = zeros(0);
    pca_R2::Float64 = 0.0;

end


@kwdef struct model_struct

    A::Matrix{Float64} = zeros(0,0)
    B::Matrix{Float64} = zeros(0,0)
    AB::Matrix{Float64} = zeros(0,0)
    Q::PDMats.PDMat{Float64, Matrix{Float64}} = init_PD(0)
    iQ::PDMats.PDMat{Float64, Matrix{Float64}} = init_PD(0)

    C::Matrix{Float64} = zeros(0,0)
    R::PDMats.PDMat{Float64, Matrix{Float64}} = init_PD(0)
    iR::PDMats.PDMat{Float64, Matrix{Float64}} = init_PD(0)

    B0::Matrix{Float64} = zeros(0,0)
    P0::PDMats.PDMat{Float64, Matrix{Float64}} = init_PD(0)
    iP0::PDMats.PDMat{Float64, Matrix{Float64}} = init_PD(0)

    CiR::Matrix{Float64} = zeros(0,0)
    CiRC::PDMats.PDMat{Float64, Matrix{Float64}} = init_PD(0)

end


function set_model(;A=A, B=B, Q=Q, C=C, R=R, B0=B0, P0=P0)::model_struct

   return model_struct(
      
    A = A,
    B = B,
    AB = [A B],
    Q = Q,
    iQ = inv(Q),

    C = C,
    R = R,
    iR = inv(R),

    B0 = B0,
    P0 = P0,
    iP0 = inv(P0),

    CiR = C'/R,
    CiRC = tol_PD(Xt_invA_X(R, C)),

    )

end



function transform_model(mdl::model_struct, W)::model_struct

    # apply transformation
    wA = W*mdl.A/W
    wB = W*mdl.B
    wQ = tol_PD(X_A_Xt(mdl.Q,W))

    wC = mdl.C/W

    wB0 = W*mdl.B0
    wP0 = tol_PD(X_A_Xt(mdl.P0, W))
    

    return model_struct(
    A = wA,
    B = wB,
    AB = [wA wB],
    Q = wQ,
    iQ = inv(wQ),

    C = wC,
    R = mdl.R,
    iR = inv(mdl.R),

    B0 = wB0,
    P0 = wP0,
    iP0 = inv(wP0),

    CiR = wC'/mdl.R,
    CiRC = tol_PD(Xt_invA_X(mdl.R, wC)),
    )

end



@kwdef struct results_struct

    #ssid
    ssid_sv::Vector{Float64} = zeros(0);

    # null models
    null_names::Vector{String} = ["cov", "AR", "enc", "encAR"]

    # logliks
    null_loglik::Vector{Float64} = zeros(4)
    null_sse_white::Vector{Float64} = zeros(4)
    null_sse_orig::Vector{Float64} = zeros(4)
    null_mse_white::Vector{Float64} = zeros(4)
    null_mse_orig::Vector{Float64} = zeros(4)

    ssid_test_R2_white::Float64 = 0.0
    ssid_test_R2_orig::Float64 = 0.0
    test_R2_white::Vector{Float64} = zeros(0)
    test_R2_orig::Vector{Float64} = zeros(0)

    ssid_fwd_R2_white::Vector{Float64} = zeros(0)
    ssid_fwd_R2_orig::Vector{Float64} = zeros(0)
    fwd_R2_white::Vector{Float64} = zeros(0)
    fwd_R2_orig::Vector{Float64} = zeros(0)

    test_sse_white::Vector{Float64} = zeros(0)
    test_sse_orig::Vector{Float64} = zeros(0)

    init_loglik::Vector{Float64} = zeros(0)
    dyn_loglik::Vector{Float64} = zeros(0)
    obs_loglik::Vector{Float64} = zeros(0)

    total_loglik::Vector{Float64} = zeros(0)
    test_loglik::Vector{Float64} = zeros(0)

    ssid_test_loglik::Float64 = 0.0
    em_test_loglik::Float64 = 0.0

    startTime_all::String = ""
    endTime_all::String = ""
    startTime_refine::String = ""
    endTime_refine::String = ""
    startTime_em::String = ""
    endTime_em::String = ""

    # initial parameters
    mdl_ssid::model_struct = model_struct()
    mdl_refine::model_struct = model_struct()
    mdl_em::model_struct = model_struct()

end




@kwdef struct estimates_struct

    # mean and cov
    pred_mean::Matrix{Float64} = zeros(0,0)
    filt_mean::Matrix{Float64} = zeros(0,0)
    smooth_mean::Matrix{Float64} = zeros(0,0)
    
    pred_cov::Vector{PDMats.PDMat{Float64, Matrix{Float64}}} = [init_PD(0)]
    pred_icov::Vector{PDMats.PDMat{Float64, Matrix{Float64}}} = [init_PD(0)]
    filt_cov::Vector{PDMats.PDMat{Float64, Matrix{Float64}}} = [init_PD(0)]
    smooth_cov::Vector{PDMats.PDMat{Float64, Matrix{Float64}}} = [init_PD(0)]
    smooth_xcov::Matrix{Float64} = zeros(0,0)

    # gains
    K::Array{Float64,3} = zeros(0,0,0)
    G::Array{Float64,3} = zeros(0,0,0)

    # transformed data
    Bu::Matrix{Float64} = zeros(0,0)
    CiRY::Matrix{Float64} = zeros(0,0)

    # temp
    xdim_temp::Vector{Float64} = zeros(0)
    ydim_temp::Vector{Float64} = zeros(0)
    xdim2_temp::Matrix{Float64} = zeros(0,0)
    x_cur::Matrix{Float64} = zeros(0,0)
    x_next::Matrix{Float64} = zeros(0,0)
    u_cur::Matrix{Float64} = zeros(0,0)
    u0_cur::Vector{Float64} = zeros(0)
    y_cur::Matrix{Float64} = zeros(0,0)


    # aggregated moments
    xx_init::PDMats.PDMat{Float64, Matrix{Float64}} = init_PD(0)
    xy_init::Matrix{Float64} = zeros(0,0)
    yy_init::Matrix{Float64} = zeros(0,0)
    n_init::Vector{Int64} = zeros(0)

    xx_dyn::Matrix{Float64} = zeros(0,0)
    xx_dyn_PD::Vector{PDMats.PDMat{Float64, Matrix{Float64}}} = [init_PD(0)]
    xy_dyn::Matrix{Float64} = zeros(0,0)
    yy_dyn::Matrix{Float64} = zeros(0,0)
    uu_dyn::PDMats.PDMat{Float64, Matrix{Float64}} = init_PD(0)
    n_dyn::Vector{Int64} = zeros(0)

    xx_obs::Matrix{Float64} = zeros(0,0)
    xx_obs_PD::Vector{PDMats.PDMat{Float64, Matrix{Float64}}} = [init_PD(0)]
    xy_obs::Matrix{Float64} = zeros(0,0)
    yy_obs::PDMats.PDMat{Float64, Matrix{Float64}} = init_PD(0)
    n_obs::Vector{Int64} = zeros(0)


    # loglik
    test_mu::Matrix{Float64} = zeros(0,0)
    test_sigma::Vector{PDMats.PDMat{Float64, Matrix{Float64}}} = [init_PD(0)]

end




function set_estimates(S)


    # initialize the moments structure with the obervables
    ywl = reshape(permutedims(S.dat.y_train, (2,3,1)), S.dat.n_steps*S.dat.n_train, S.dat.y_dim);
    uwl = reshape(permutedims(S.dat.u_train[:,1:end-1,:], (2,3,1)), (S.dat.n_steps-1)*S.dat.n_train, S.dat.u_dim);
    u0wl = reshape(permutedims(S.dat.u0_train, (2,1)), S.dat.n_train, S.dat.u0_dim);

    est = estimates_struct(
        
        pred_mean = zeros(S.dat.x_dim, S.dat.n_steps),
        filt_mean = zeros(S.dat.x_dim, S.dat.n_steps),
        smooth_mean = zeros(S.dat.x_dim, S.dat.n_steps),

        pred_cov = [init_PD(S.dat.x_dim) for _ in 1:S.dat.n_steps],
        pred_icov = [init_PD(S.dat.x_dim) for _ in 1:S.dat.n_steps],
        filt_cov = [init_PD(S.dat.x_dim) for _ in 1:S.dat.n_steps],
        smooth_cov = [init_PD(S.dat.x_dim) for _ in 1:S.dat.n_steps],
        smooth_xcov = zeros(S.dat.x_dim, S.dat.x_dim),

        K = zeros(S.dat.x_dim, S.dat.y_dim, S.dat.n_steps),
        G = zeros(S.dat.x_dim, S.dat.x_dim, S.dat.n_steps),

        Bu = zeros(S.dat.x_dim, S.dat.n_steps),
        CiRY = zeros(S.dat.x_dim, S.dat.n_steps),

        xdim_temp = zeros(S.dat.x_dim),
        xdim2_temp = zeros(S.dat.x_dim, S.dat.x_dim),
        ydim_temp = zeros(S.dat.y_dim),

        x_cur = zeros(S.dat.x_dim, S.dat.n_steps-1),
        x_next  = zeros(S.dat.x_dim, S.dat.n_steps-1),
        u_cur  = zeros(S.dat.u_dim, S.dat.n_steps-1),
        u0_cur  = zeros(S.dat.u0_dim),
        y_cur = zeros(S.dat.y_dim, S.dat.n_steps),
        
        xx_init = tol_PD(u0wl' * u0wl),
        xy_init = zeros(S.dat.u0_dim, S.dat.x_dim),
        yy_init = zeros(S.dat.x_dim, S.dat.x_dim),
        n_init = zeros(1),
    
        xx_dyn = zeros(S.dat.x_dim + S.dat.u_dim, S.dat.x_dim + S.dat.u_dim),
        xx_dyn_PD = [init_PD(S.dat.x_dim + S.dat.u_dim)],
        xy_dyn = zeros(S.dat.x_dim + S.dat.u_dim, S.dat.x_dim),
        yy_dyn = zeros(S.dat.x_dim, S.dat.x_dim),
        uu_dyn = tol_PD(uwl' * uwl),
        n_dyn = zeros(1),
    
        xx_obs = zeros(S.dat.x_dim, S.dat.x_dim),
        xx_obs_PD = [init_PD(S.dat.x_dim)],
        xy_obs = zeros(S.dat.x_dim, S.dat.y_dim),
        yy_obs = tol_PD(ywl' * ywl),
        n_obs = zeros(1),

        test_mu = zeros(S.dat.y_dim, S.dat.n_steps),
        test_sigma = [init_PD(S.dat.y_dim)],

    )


    return est

end



@kwdef struct core_struct

    prm::param_struct
    dat::data_struct
    res::results_struct
    est::estimates_struct
    mdl::model_struct

end



@kwdef struct post_all

    pred_mean::Array{Float64} = zeros(0,0,0)
    filt_mean::Array{Float64} = zeros(0,0,0)
    smooth_mean::Array{Float64} = zeros(0,0,0)

    pred_cov::Vector{Vector{PDMats.PDMat{Float64, Matrix{Float64}}}} = [[init_PD(0)]]
    filt_cov::Vector{Vector{PDMats.PDMat{Float64, Matrix{Float64}}}} = [init_PD(0)]
    smooth_cov::Vector{Vector{PDMats.PDMat{Float64, Matrix{Float64}}}} = [[init_PD(0)]]


    obs_white_y::Array{Float64} = zeros(0,0,0)
    pred_white_y::Array{Float64} = zeros(0,0,0)
    filt_white_y::Array{Float64} = zeros(0,0,0)
    smooth_white_y::Array{Float64} = zeros(0,0,0)
    
    obs_orig_y::Array{Float64} = zeros(0,0,0)
    pred_orig_y::Array{Float64} = zeros(0,0,0)
    filt_orig_y::Array{Float64} = zeros(0,0,0)
    smooth_orig_y::Array{Float64} = zeros(0,0,0)

    sse_white::Vector{Float64} = [0.0]
    sse_orig::Vector{Float64} = [0.0]

    sse_fwd_white::Vector{Float64} = zeros(26)
    sse_fwd_orig::Vector{Float64} = zeros(26)

end



@kwdef struct post_mean

    pred_mean::Array{Float64} = zeros(0,0,0)
    filt_mean::Array{Float64} = zeros(0,0,0)
    smooth_mean::Array{Float64} = zeros(0,0,0)

end



@kwdef struct post_sse

    sse_white::Vector{Float64} = [0.0]
    sse_orig::Vector{Float64} = [0.0]

    sse_fwd_white::Vector{Float64} = zeros(26)
    sse_fwd_orig::Vector{Float64} = zeros(26)

end

