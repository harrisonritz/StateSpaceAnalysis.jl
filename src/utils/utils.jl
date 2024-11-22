# utility functions



# tol_PD =============================
function tol_PD(A_sym::Union{Symmetric, Hermitian, PDMat}; tol=1e-6)::PDMat
    """
        tol_PD(A_sym::Union{Symmetric, Hermitian, PDMat}; tol=1e-6) -> PDMat

    Adjusts the eigenvalues of a positive definite matrix to ensure numerical stability.

    # Arguments
    - `A_sym::Union{Symmetric, Hermitian, PDMat}`: A symmetric, Hermitian, or positive definite matrix.
    - `tol::Float64`: Tolerance level for adjusting the eigenvalues. Default is `1e-6`.

    # Returns
    - `PDMat`: A positive definite matrix with adjusted eigenvalues.

    # Description
    This function takes a symmetric, Hermitian, or positive definite matrix `A_sym` and adjusts its eigenvalues to ensure numerical stability. The eigenvalues are scaled and shifted based on the provided tolerance `tol`. The resulting matrix is guaranteed to be positive definite.
    """

    l, Q = eigen!(A_sym);    

    l_r = max.(l ./ l[end], 0.0);
    newl =  (l[end] - l[end]*tol).*l_r .+ l[end]*tol;
    return PDMat(X_A_Xt(PDiagMat(newl), Q));

end

tol_PD(A::Matrix; tol=1e-6)::PDMat = tol_PD(hermitianpart(A); tol=tol);


# tol_PSD =============================
function tol_PSD(A_sym::Union{Symmetric, Hermitian, PDMat})::Hermitian

    l, Q = eigen!(A_sym);
    return X_A_Xt(PDiagMat(max.(l, 0.0)), Q)

end

tol_PSD(A::Matrix)::Hermitian = tol_PSD(hermitianpart(A))::Hermitian;




# diag_PD =============================
function diag_PD(A; tol=1e-6)
    # this should be improved to match tol_PD
    # however, don't use diagonal noise

    return PDiagMat(max.(diag(A), tol));

end


# format_noise =============================
function format_noise(X, type; tol=1e-6)

    if type == "identity"

        Xf = I(size(X,1));

    elseif type == "diagonal"

        Xf = diag_PD(X; tol=tol);

    elseif type == "full"

        Xf = tol_PD(X; tol=tol);

    else

        error("type not recognized")

    end

    return Xf

end



# random parameters

function init_param_rand(S)




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



# misc =============================
init_PD(d) = PDMat(diagm(ones(d)));

init_PSD(d) = Hermitian(diagm(ones(d)));

zsel(x,sel) =  (x[sel] .- mean(x[sel])) ./ std(x[sel]);

zsel_tall(x,sel) =  ((x .- mean(x[sel])) ./ std(x[sel])).*sel;

zdim(x;dims=1) = (x .- mean(x, dims=dims)) ./ std(x, dims=dims);

sumsqr(x) = sum(x.*x);

split_list(x) = split(x, "@");

demix(S, y) = S.dat.W' * (y .- S.dat.mu);
remix(S, y) = (S.dat.W * y) .+ S.dat.mu;
