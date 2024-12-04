# functions for generating parameters and data


# init random parameters
function gen_rand_params(S)

    A = Matrix(Diagonal(rand(S.dat.x_dim)));
    B = randn(S.dat.x_dim, S.dat.u_dim);
    Q = tol_PD(randn(S.dat.x_dim, S.dat.x_dim));

    C = randn(S.dat.y_dim, S.dat.x_dim);
    R = tol_PD(randn(S.dat.y_dim, S.dat.y_dim));

    B0 = randn(S.dat.x_dim, S.dat.u0_dim);
    P0 = tol_PD(randn(S.dat.x_dim, S.dat.x_dim));

    @reset S.mdl = set_model(;A=A, B=B, Q=Q, C=C, R=R, B0=B0, P0=P0);

    return S

end



# generate random data
function generate_ssm_trials(A, B, Q, C, R, B0, P0, u, u0, n_steps, n_trials)
    """ Generate data from a linear dynamical system
    Args:
        A : state transition matrix (dim_x, dim_x)
        B : control matrix (dim_x, dim_u)
        Q : state noise covariance (dim_x, dim_x)
        C : observation matrix (dim_y, dim_x)
        R : observation noise covariance (dim_y, dim_y)
        u : inputs (dim_u, n_steps, n_trials)
        u0 : initial inputs (dim_u x n_trials)
        n_steps : number of time steps
        n_trials : number of trials
    Returns:
        x : latent states (dim_x, n_steps, n_trials)
        y : observations (dim_y, n_steps, n_trials)
    """

    # get dimensions
    dim_x = size(A, 1)
    dim_y = size(C, 1)
    dim_u = size(B, 2)

    # make sure if u are not None then they have the right shape
    if u !== nothing
        @assert size(u, 1) == dim_u
        @assert size(u, 2) == n_steps
        @assert size(u, 3) == n_trials
    end

    # initialize latent states
    x = zeros(dim_x, n_steps, n_trials)
    y = zeros(dim_y, n_steps, n_trials)

    for tt in axes(y,3)

        # initialize latent state
        x[:,1,tt] = rand(MvNormal(B0*u0[:,tt], P0))
  
        # generate latent states
        for nn in axes(x,2)[2:end]
            x[:, nn, tt] = A*x[:, nn-1, tt] + B*u[:,nn-1,tt] + rand(MvNormal(zeros(dim_x), Q))
        end

        # generate observations
        for nn in axes(y,2)
            y[:, nn, tt] = C * x[:, nn, tt] + rand(MvNormal(zeros(dim_y), R))
        end

    end

    # return x and y
    return x, y

end


function generate_ssm_trials(S::core_struct, u, u0, n_steps, n_trials)
    """ Generate data from a linear dynamical system
    Args:
        S : core struct
        u : inputs (dim_u, n_steps, n_trials)
        u0 : initial inputs (dim_u x n_trials)
        n_steps : number of time steps
        n_trials : number of trials
    Returns:
        x : latent states (dim_x, n_steps, n_trials)
        y : observations (dim_y, n_steps, n_trials)
    """

    # get dimensions
    dim_x = size(S.mdl.A, 1)
    dim_y = size(S.mdl.C, 1)
    dim_u = size(S.mdl.B, 2)

    # make sure if u are not None then they have the right shape
    if u !== nothing
        @assert size(u, 1) == dim_u
        @assert size(u, 2) == n_steps
        @assert size(u, 3) == n_trials
    end

    # initialize latent states
    x = zeros(dim_x, n_steps, n_trials)
    y = zeros(dim_y, n_steps, n_trials)

    for tt in axes(y,3)

        # initialize latent state
        x[:,1,tt] = rand(MvNormal(S.mdl.B0*u0[:,tt], S.mdl.P0))
  
        # generate latent states
        for nn in axes(x,2)[2:end]
            x[:, nn, tt] = S.mdl.A*x[:, nn-1, tt] + S.mdl.B*u[:,nn-1,tt] + rand(MvNormal(zeros(dim_x), S.mdl.Q))
        end

        # generate observations
        for nn in axes(y,2)
            y[:, nn, tt] = S.mdl.C * x[:, nn, tt] + rand(MvNormal(zeros(dim_y), S.mdl.R))
        end

    end

    # return x and y
    return x, y

end

