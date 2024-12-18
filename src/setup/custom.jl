# This file contains the setup functions for the SSID algorithm.


# function to assign arguments
# function  assign_arguments(S, conds)
#     @reset S.dat.pt = conds[1];
#     @reset S.dat.x_dim = conds[2];

#     return S

# end



# select trials
function select_trials(S)

    # custom selection
    if S.prm.do_trial_sel
        println("custom trial sel")
        @reset S.dat.sel_trial =    vec(S.dat.trial["acc"] .== 1) .& 
                                    vec(S.dat.trial["prevAcc"] .== 1) .& 
                                    vec(isfinite.(S.dat.trial["RT"])) .& 
                                    vec(isfinite.(S.dat.trial["block"])); # current and previous accurate
    else
        println("minimal trial sel")
        @reset S.dat.sel_trial =    vec(isfinite.(S.dat.trial["RT"])) .& 
                                    vec(isfinite.(S.dat.trial["block"]));
    end


    # select test trials
    unique_blocks = unique(S.dat.trial["block"][:,1]);
    test_block = unique_blocks[round(Int64, length(unique_blocks)/2)];

    @reset S.dat.sel_train = S.dat.sel_trial .& vec(S.dat.trial["block"]  .!= test_block);
    @reset S.dat.sel_test = S.dat.sel_trial .& vec(S.dat.trial["block"] .== test_block);
    

    return S

end



# scale inputs
function scale_input(u,sel)

    return zsel(vec(u), sel);
   
end



# format inputs with basis functions
function create_input_basis(S, n_trials)

    if isempty(S.dat.basis_name)
        println("setting default basis")
        @reset S.dat.basis_name = "bspline";
    end

    println("basis: $(S.dat.basis_name)")


    if S.dat.basis_name == "bspline"


        # set up basis
        if S.dat.n_splines > 0

            @reset S.dat.n_bases = S.dat.n_splines;
            basis = averagebasis(4, LinRange(1, S.dat.n_steps, S.dat.n_bases));
            pred_basis = ["spline" for _ in 1:S.dat.n_bases];

        elseif S.dat.spline_gap > 0

            @reset S.dat.n_bases = round(S.dat.n_steps/S.dat.spline_gap);
            basis = averagebasis(4, LinRange(1, S.dat.n_steps, S.dat.n_bases));
            pred_basis = ["spline" for _ in 1:S.dat.n_bases];

        else
            error("n_splines or spline_gap must be greater than 0")
        end
        @reset S.dat.u_dim = S.dat.n_misc + S.dat.n_bases + (S.dat.n_bases*S.dat.n_pred);

        # construct basis
        u = zeros(S.dat.u_dim, S.dat.n_steps, n_trials); 

        normalize_col(A,d=1) = A ./ (sqrt.(sum(abs2,A,dims=d)))
        for tt in axes(u,2)
            bs = bsplines(basis, tt);
            u[collect(axes(bs,1)),tt,:] .= S.dat.norm_basis ? normalize_col(collect(bs)) : collect(bs);
        end

        if S.dat.norm_basis
            println("normalized basis")
        else
            println("unnormalized basis")
        end

        println("n bases: $(S.dat.n_bases), breakpoints: $(round.(breakpoints(basis),sigdigits=4))")


    else
        error("basis not implemented")
    end

    return u, deepcopy(S.dat.n_bases), deepcopy(S.dat.u_dim), pred_basis

end




# transform observations (typically PCA)
function transform_observations(S, y_long)


    if  uppercase(S.prm.y_transform) == "PCA"

        # fit PCA
        pca = fit(PCA, y_long, pratio=S.prm.PCA_ratio, maxoutdim=S.prm.PCA_maxdim); # ==== do PCA

        # enforce sign convention (largest component is positive)
        W = pca.proj;
        for cc in axes(W,2)
            W[:,cc] .*= sign(W[findmax(abs.(W[:,cc]))[2],cc]);
        end

        # save
        @reset S.dat.W = W;
        @reset S.dat.mu = pca.mean;
        @reset S.dat.pca_R2 = pca.tprinvar/pca.tvar;
        @reset S.dat.y_dim = size(S.dat.W, 2);


        println("\n========== PCA with dim=$(S.dat.y_dim) ==========")
        println("variance included: $(S.dat.pca_R2)");
        println("eigenvalues: $(round.(principalvars(pca)'))");


    elseif  uppercase(S.prm.y_transform) == "WHITEN"

        pca = fit(Whitening, y_long, regcoef=1e-6); # ==== do whitening

        @reset S.dat.W = pca.W;
        @reset S.dat.mu = pca.mean;
        @reset S.dat.pca_R2 = 1.0;
        @reset S.dat.y_dim = deepcopy(S.dat.n_chans);

        println("\n========== Whitening with dim=$(S.dat.y_dim) ==========")


    elseif  uppercase(S.prm.y_transform) == "NONE"

        @reset S.dat.W = Matrix(1.0I(S.dat.n_chans));
        @reset S.dat.mu = vec(mean(y_long, dims=2)); # center observations
        @reset S.dat.pca_R2 = 1.0;
        @reset S.dat.y_dim = deepcopy(S.dat.n_chans);

        println("\n========== No Transform with dim=$(S.dat.y_dim) ==========")


    else
        error("invalid y_transform")
    end
    println("========================================\n")


    return S

end





# format the B matrix for SSID
function format_B_preSSID(S)
    # Extract a subset of predictors columns for SSID.
    # This helps to keep SSID well-posed (it does best with sparse inputs)
    # e.g., we only use the second basis function for each predictor (to avoid collinearity with the initial condition)
    # write your own functions here

    if (S.dat.basis_name == "bspline")

        println("init for bspline")
        u = deepcopy(S.dat.u_train[2:S.dat.n_bases:end,:,:]);

    else

        u = deepcopy(S.dat.u_train);

    end

    println("u size: ", size(u))
    return u

end





function format_B_postSSID(S, sys)
    # after SSID, assign the estimated B columns to the rest of the matrix
    # e.g., we assign the estimated B columns from the second basis function to the rest of the matrix
    # write your own functions here

    if (S.dat.basis_name == "bspline") 

        sysB = deepcopy(sys.B);

        Bd = zeros(S.dat.x_dim, S.dat.u_dim);
        for ii in 1:S.dat.n_bases
            Bd[:,ii:S.dat.n_bases:end] = sysB/S.dat.n_bases;
        end
    
    else

        Bd = deepcopy(sys.B);

    end

    return Bd

end


