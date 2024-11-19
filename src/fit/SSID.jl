

 


function fit_SSID(S)
    """
    run_SSID(S::Structure)

    Runs the SSID algorithm on the input `S`.

    # Arguments
    - `S::SomeType`: The input data on which the SSID algorithm will be executed.

    # Returns
    - The result of the SSID algorithm.

    # Example
    ```julia
    result = run_SSID(S)
    ```
    """ 


    # reformat data ==================================================
    y = deepcopy(S.dat.y_train);
    y_long = reshape(y, size(y,1), size(y,2)*size(y,3));

    # use subset of predictors to keep SSID well-posed + reduce computation
    u = format_B_preSSID(S);
    u_long = reshape(u, size(u,1), size(u,2)*size(u,3));

    # make system
    id = iddata(y_long, u_long, S.dat.dt);
    println(id);


    # run subspace identification ==================================================
    # modifed version of subspaceid from ControlSystemIdentification.jl

    @views @inbounds @inline sys = subspaceid_SSA(id, S.dat.x_dim; 
                                            r=S.prm.ssid_lag,
                                            u_orig=u_long,
                                            stable=true,
                                            verbose=true, 
                                            scaleU=false, 
                                            zeroD=true,
                                            Aestimator = ridge_estimator,
                                            Bestimator = ridge_estimator,
                                            W=S.prm.ssid_type);

    
    # format system ==================================================
    # dynamics terms
    Ad = deepcopy(sys.A);

    # reformat B matrix
    Bd = format_B_postSSID(S, sys);

    Cd = deepcopy(sys.C);

    x0 = Cd\y[:,1,:];
    B0d = x0/S.dat.u0_train;

    # noise terms
    Qd = format_noise(sys.Q, S.prm.Q_refine_type);
    Rd = format_noise(sys.R, S.prm.R_refine_type);
    P0d = format_noise(sys.P, S.prm.P0_refine_type);
      

    # save model ==================================================
    @reset S.mdl = set_model(
                            A = Ad,
                            B = Bd,
                            Q = Qd,
                            C = Cd,
                            R = Rd,
                            B0 = B0d,
                            P0 = P0d,
                            );

    @reset S.res.mdl_ssid = deepcopy(S.mdl)
    @reset S.res.ssid_sv = sys.s.S;


    return S
  
end




# use ridge regression for SSID
function ridge_estimator(x::AbstractArray{Float64},y::AbstractArray{Float64})
    
    p = size(x,2);

    xr = [x; sqrt(1e-3)I(p)];
    yr = [y; zeros(p, size(y,2))];
    br = xr\yr;

    return br;
end


function ridge_estimator(x::QRPivoted,y::AbstractArray{Float64})

    diagR = Diagonal(x.R);
    x.R .+= sqrt.(diagR.^2 + 1e-3I(size(x.R,2))) - diagR;
    br = x \ y;
    
    return br;

end




function subspaceid_SSA(
    data::InputOutputData,
    nx = :auto;
    u_orig=Matrix{Float64}(undef, 0, 0),
    verbose = false,
    r = nx === :auto ? min(length(data) ÷ 20, 50) : 2nx + 10, # the maximal prediction horizon used
    s1 = r, # number of past outputs
    s2 = r, # number of past inputs
    γ = nothing, # discarded, aids switching from n4sid
    W = :CVA,
    zeroD = false,
    stable = true, 
    focus = :prediction,
    svd::F1 = svd!,
    scaleU = true,
    Aestimator::F2 = \,
    Bestimator::F3 = \,
    weights = nothing
) where {F1,F2,F3}

    nx !== :auto && r < nx && throw(ArgumentError("r must be at least nx"))
    y, u = transpose(copy(output(data))), transpose(copy(input(data)))
    if isempty(u_orig)
        println("u_orig is empty")
        u_orig = deepcopy(u)
    end

    # println("size y = $(size(y)), size u = $(size(u))")
    if scaleU
        CU = std(u, dims=1)
        u ./= CU
    end
    t, p = size(y, 1), size(y, 2)
    m = size(u, 2)
    t0 = max(s1,s2)+1
    s = s1*p + s2*m
    N = t - r + 1 - t0

    @views @inbounds function hankel(u::AbstractArray, t0, r)
        d = size(u, 2)
        H = zeros(eltype(u), r * d, N)
        for ri = 1:r, Ni = 1:N
            H[(ri-1)*d+1:ri*d, Ni] = u[t0+ri+Ni-2, :] # TODO: should start at t0
        end
        H
    end

    # 1. Form G  (10.103). (10.100). (10.106). (10.114). and (10.108).

    Y = hankel(y, t0, r) # these go forward in time
    U = hankel(u, t0, r) # these go forward in time
    # @assert all(!iszero, Y) # to be turned off later
    # @assert all(!iszero, U) # to be turned off later
    @assert size(Y) == (r*p, N)
    @assert size(U) == (r*m, N)
    φs(t) = [ # 10.114
        y[t-1:-1:t-s1, :] |> vec # QUESTION: not clear if vec here or not, Φ should become s × N, should maybe be transpose before vec, but does not appear to matter
        u[t-1:-1:t-s2, :] |> vec
    ]

    Φ = reduce(hcat, [φs(t) for t ∈ t0:t0+N-1]) # 10.108. Note, t can not start at 1 as in the book since that would access invalid indices for u/y. At time t=t0, φs(t0-1) is the first "past" value
    @assert size(Φ) == (s, N)

    println("ssid lq...")
    UΦY = [U; Φ; Y]
    l = lq!(UΦY)
    L = l.L
    if W ∈ (:MOESP, :N4SID)
        Q = Matrix(l.Q) # (pr+mr+s × N) but we have adjusted effective N
    end
    l = nothing # free memory

    # @assert size(Q) == (p*r+m*r+s, N) "size(Q) == $(size(Q)), if this fails, you may need to lower the prediction horizon r which is currently set to $r"
    Uinds = 1:size(U,1)
    Φinds = (1:size(Φ,1)) .+ Uinds[end]
    Yinds = (1:size(Y,1)) .+ (Uinds[end]+s)
    @assert Yinds[end] == p*r+m*r+s

    L1 = L[Uinds, Uinds]
    L2 = L[s1*p+(r+s2)*m+1:end, 1:s1*p+(r+s2)*m+p]
   

    # 2. Select weighting matrices W1 (rp × rp)
    # and W2 (p*s1 + m*s2 × α) = (s × α)
    # @assert size(Ĝ, 1) == r*p
    if W ∈ (:MOESP, :N4SID)

        L21 = L[Φinds, Uinds]
        L22 = L[Φinds, Φinds]
        L32 = L[Yinds, Φinds]
       
        Q1 = Q[Uinds, :]
        Q2 = Q[Φinds, :]
        Ĝ = L32*(L22\[L21 L22])*[Q1; Q2] # this G is used for N4SID weight, but also to form Yh for all methods

        if W === :MOESP
            W1 = I
            # W2 = 1/N * (Φ*ΠUt*Φ')\Φ*ΠUt
            G = L32*Q2 #* 1/N# QUESTION: N does not appear to matter here
        elseif W === :N4SID
            W1 = I
            # W2 = 1/N * (Φ*ΠUt*Φ')\Φ
            G = deepcopy(Ĝ) #* 1/N
        end

    elseif W ∈ (:IVM, :CVA)

        if W === :IVM

            UY = [U; Y]
            Yinds = (1:size(Y,1)) .+ size(U,1)
            YΠUt = proj_hr(UY, Yinds)
            
            G = YΠUt*Φ' #* 1/N # 10.109, pr×s # N does not matter here
            @assert size(G) == (p*r, s)
            W1 = sqrt(Symmetric(pinv(inv(N) * (YΠUt*Y')))) |> real
            W2 = sqrt(Symmetric(pinv(inv(N) * Φ*Φ'))) |> real
            G = W1*G*W2
            @assert size(G, 1) == r*p

        elseif W === :CVA        

            L32 = L[Yinds, Φinds]
            W1 = L[Yinds,[Φinds; Yinds]]

            println("ssid svd...")
            ull1,sll1 = svd!(W1)
            sll1 = Diagonal(sll1[1:r*p])
            # Or,Sn = svd(pinv(sll1)*ull1'*L32)
            cva_svd = svd!(sll1\(ull1'*L32))
            Or = ull1*sll1*cva_svd.U
            # ΦΠUt = proj(Φ, U)
            # W1 = pinv(sqrt(1/N * (YΠUt*Y'))) |> real
            # W2 = pinv(sqrt(1/N * ΦΠUt*Φ')) |> real
            # G = W1*G*W2

        end

        # @assert size(W1) == (r*p, r*p)
        # @assert size(W2, 1) == p*s1 + m*s2

    else
        throw(ArgumentError("Unknown choice of W"))
    end

    # 3. Select R and define Or = W1\U1*R
    sv = W === :CVA ? svd!(L32) : svd!(G)
    if nx === :auto
        nx = sum(sv.S .> sqrt(sv.S[1] * sv.S[end]))
        verbose && @info "Choosing order $nx"
    end
    n = nx
    S1 = sv.S[1:n]
    R = Diagonal(sqrt.(S1))
    if W !== :CVA
        U1 = sv.U[:, 1:n]
        V1 = sv.V[:, 1:n]
        Or = W1\(U1*R)
    end
    
    fve = sum(S1) / sum(sv.S)
    verbose && @info "Fraction of variance explained: $(fve)"

    C = Or[1:p, 1:n]
    A = Aestimator(Or[1:p*(r-1), 1:n] , Or[p+1:p*r, 1:n])
    if !all(e->abs(e)<=1, eigvals(A))
        if stable
            verbose && @info "A matrix unstable -- stabilizing by reflection"
            A = reflectd(A)
        else
            verbose && @info "A matrix unstable -- NOT stabilizing"
        end
    end




    println("estimating noise & gain ...")
    @views @inbounds @inline P, K, Qc, Rc, Sc = find_PK_SSA(L1,L2,Or,n,p,m,r,s1,s2,A,C)


    println("estimating B & D ...")
    pred_K = (focus === :prediction)*K
    ut = u_orig
    ut = transpose(u)
    mt = size(u_orig, 1)
    yt = transpose(y)

    @views @inbounds @inline B, D, x0 = find_BD_SSA(A, pred_K, C, u_orig, yt, mt, zeroD, Bestimator, weights)
    



    # TODO: iterate find C/D and find B/D a couple of times

    if scaleU
        B ./= CU
        D ./= CU
    end


    
    N4SIDStateSpace(ss(A,  B,  C,  D, data.Ts), Qc,Rc,Sc,K,P,x0,sv,fve)

end




function find_BD_SSA(A,K,C,U,Y,m, zeroD=false, estimator=\, weights=nothing)
    T = eltype(A)
    nx = size(A, 1)
    p = size(C, 1)
    N = size(U, 2)
    A = A-K*C
    y_hat = lsim(ss(A,K,C,0,1), Y)[1] # innovation sequence
    φB = zeros(Float64, p, N, m*nx)
    @inbounds @views for (j,k) in Iterators.product(1:nx, 1:m)
        E = zeros(nx)
        E[j] = 1
        fsys = ss(A, E, C, 0, 1)
        u = U[k:k,:]
        uf = lsim(fsys, u)[1]
        r = (k-1)*nx+j
        @inbounds φB[:,:,r] .= uf 
    end
    φx0 = zeros(p, N, nx)
    x0u = zeros(1, N)
    @inbounds @views for (j,k) in Iterators.product(1:nx, 1:1)
        E = zeros(nx)
        x0 = zeros(nx); x0[j] = 1
        fsys = ss(A, E, C, 0, 1)
        uf = lsim(fsys, x0u; x0)[1]
        r = (k-1)*nx+j
        φx0[:,:,r] = uf 
    end
    if !zeroD
        φD = zeros(Float64, p, N, m*p)
        for (j,k) in Iterators.product(1:p, 1:m)
            E = zeros(p)
            E[j] = 1
            fsys = ss(E, 1)
            u = U[k:k,:]
            uf = lsim(fsys, u)[1]
            r = (k-1)*p+j
            φD[:,:,r] = uf 
        end
    end

    if zeroD

        φ3 = zeros(Float64, p, N, (m+1)*nx);
        @inbounds φ3[:,:,1:(m*nx)] .= φB;
        @inbounds φ3[:,:,((m*nx)+1):end] .= φx0;
        # φ3 = cat(φB, φx0, dims=Val(3));
        
        φ = reshape(φ3, (p*N, (m+1)*nx));
        φqr = qr!(φ, ColumnNorm());

        φ = nothing;
        φ3 = nothing;

    else
        φ3 = cat(φB, φx0, φD, dims=Val(3))
        φqr = reshape(φ3, p*N, :)
    end

    # @inbounds φ3 = zeroD ? cat(φB, φx0, dims=Val(3)) : cat(φB, φx0, φD, dims=Val(3))
    # φ4 = permutedims(φ3, (1,3,2))

    if weights === nothing
        BD = estimator(φqr, vec(Y .- y_hat))
    else
        BD = estimator(φqr, vec(Y .- y_hat), weights)
    end
    B = copy(reshape(BD[1:m*nx], nx, m))
    x0 = BD[m*nx .+ (1:nx)]
    if zeroD
        D = zeros(T, p, m)
    else
        D = copy(reshape(BD[end-p*m+1:end], p, m))
        B .+= K*D
    end
    B,D,x0
end

function find_BDf(A, C, U, Y, λ, zeroD, Bestimator, estimate_x0)
    nx = size(A,1)
    ny, nw = size(Y)
    nu = size(U, 1)
    if estimate_x0
        ue = [U; transpose(λ)] # Form "extended input"
        nup1 = nu + 1
    else
        ue = U
        nup1 = nu
    end

    sys0 = ss(A,I(nx),C,0) 
    F = evalfr2(sys0, λ)
    # Form kron matrices
    if zeroD      
        AA = similar(U, nw*ny, nup1*nx)
        for i in 1:nw
            r = ny*(i-1) + 1:ny*i
            for j in 1:nup1
                @views AA[r, ((j-1)nx) + 1:j*nx] .= ue[j, i] .* (F[:, :, i])
            end
        end
    else
        AA = similar(U, nw*ny, nup1*nx+nu*ny) 
        for i in 1:nw
            r = (ny*(i-1) + 1) : ny*i
            for j in 1:nup1
                @views AA[r, (j-1)nx + 1:j*nx] .= ue[j, i] .* (F[:, :, i])
            end
            for j in 1:nu
                @views AA[r, nup1*nx + (j-1)ny + 1:nup1*nx+ny*j] = ue[j, i] * I(ny)
            end
        end
    end
    vy = vec(Y)
    YY = [real(vy); imag(vy)]
    AAAA = [real(AA); imag(AA)]
    BD = Bestimator(AAAA, YY)
    e = YY - AAAA*BD
    B = reshape(BD[1:nx*nup1], nx, :)
    D = zeroD ? zeros(eltype(B), ny, nu) : reshape(BD[nx*nup1+1:end], ny, nu)
    if estimate_x0
        x0 = B[:, end]
        B = B[:, 1:end-1]
    else
        x0 = zeros(eltype(B), nx)
    end
    return B, D, x0, e
end

function find_CDf(A, B, U, Y, λ, x0, zeroD, Bestimator, estimate_x0)
    nx = size(A,1)
    ny, nw = size(Y)
    nu = size(U, 1)
    if estimate_x0
        Ue = [U; transpose(λ)] # Form "extended input"
        Bx0 = [B x0]
    else
        Ue = U
        Bx0 = B
    end

    sys0 = ss(A,Bx0,I(nx),0)
    F = evalfr2(sys0, λ, Ue)
    # Form kron matrices
    if zeroD      
        AA = F
    else
        AA = [F; U]
    end


    YY = [real(transpose(Y)); imag(transpose(Y))]
    AAAA = [real(AA) imag(AA)]
    CD = Bestimator(transpose(AAAA), YY) |> transpose
    e = YY - transpose(AAAA)*transpose(CD)
    C = CD[:, 1:nx]
    D = zeroD ? zeros(eltype(C), ny, nu) : CD[:, nx+1:end]
    return C, D, e
end

function proj_hr(UY, Yinds)
    # UY = [U; Yi]
    l = lq!(UY)
    L = l.L
    Q = Matrix(l.Q) # (pr+mr+s × N) but we have adjusted effective N
    # Uinds = 1:size(U,1)
    # Yinds = (1:size(Yi,1)) .+ Uinds[end]
    # if Yi === Y
        # @assert size(Q) == (p*r+m*r, N) "size(Q) == $(size(Q))"
        # @assert Yinds[end] == p*r+m*r
    # end
    L22 = L[Yinds, Yinds]
    Q2 = Q[Yinds, :]
    L22*Q2
end





function find_PK_SSA(L1,L2,Or,n,p,m,r,s1,s2,A,C)
    X1 = L2[p+1:r*p, 1:m*(s2+r)+p*s1+p]
    X2 = [L2[1:r*p,1:m*(s2+r)+p*s1] zeros(r*p,p)]
    vl = [Or[1:(r-1)*p, 1:n]\X1; L2[1:p, 1:m*(s2+r)+p*s1+p]]
    hl = [Or[:,1:n]\X2 ; [L1 zeros(m*r,(m*s2+p*s1)+p)]]
    
    K0 = vl*pinv(hl)
    W = (vl - K0*hl)*(vl-K0*hl)'
    
    Q = W[1:n,1:n] |> Hermitian
    S = W[1:n,n+1:n+p]
    R = W[n+1:n+p,n+1:n+p] |> Hermitian
    
    local P, K
    try
        a = 1/sqrt(mean(abs, Q)*mean(abs, R)) # scaling for better numerics in ared
        P, _, Kt, _ = ControlSystemIdentification.MatrixEquations.ared(copy(A'), copy(C'), a*R, a*Q, a*S)
        K = Kt' |> copy
    catch e
        @error "Failed to estimate kalman gain, got error" 
        P = I(n)
        K = zeros(n, p)
    end
 
    P, K, Q, R, S
end

function reflectd(x)
    a = abs(x)
    a < .9999 && return oftype(cis(angle(x)),x)
    (.9999)/a * cis(angle(x))
end

function reflectd(A::AbstractMatrix)
    D,V = eigen(A)
    D = reflectd.(D)
    A2 = V*Diagonal(D)/V
    if eltype(A) <: Real
        return real(A2)
    end
    A2
end


