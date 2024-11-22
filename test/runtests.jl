
# packages
using StateSpaceAnalysis
using Test
using Accessors
using Revise



@testset "StateSpaceAnalysis.jl" begin


    # Test for read_args function
    @testset "read_args" begin
        S = core_struct(
            prm=param_struct(root_path="", save_path=""), 
            dat=data_struct(),
            res=results_struct(),
            est=estimates_struct(),
            mdl=model_struct(),
            );
        arg_in = ["1", "true"]
        S = read_args(S, arg_in)
        @test !isempty(S.prm.save_name)
        @test S.prm.do_fast == true || S.prm.do_fast == false
    end



    # Test for setup_dir function
    @testset "setup_dir" begin
        S = core_struct(
            prm=param_struct(root_path="", save_path=""), 
            dat=data_struct(),
            res=results_struct(),
            est=estimates_struct(),
            mdl=model_struct(),
            );
        @reset S.prm.save_path = "test"
        @reset S.prm.model_name = "test_model"
        StateSpaceAnalysis.setup_path(S)
        @test isdir("test/fit-results/figures/test_model")
    end



    # TODO: working here!
    # Test for load_data function
    @testset "load_data" begin
        S = core_struct(
            prm=param_struct(
                load_name="example",
                load_path="example/example-data"), 
            dat=data_struct(
                sel_event = 2:2
            ),
            res=results_struct(),
            est=estimates_struct(),
            mdl=model_struct(),
            );
        @reset S = StateSpaceAnalysis.load_data(S)
        @test !isempty(S.dat.y_train_orig)
    end



    # Test for build_inputs function
    @testset "build_inputs" begin
        S = core_struct(
            prm=param_struct(
                load_name="example",
                load_path="example/example-data"), 
            dat=data_struct(),
            res=results_struct(),
            est=estimates_struct(),
            mdl=model_struct(),
            );
        S = StateSpaceAnalysis.load_data(S);
        S = build_inputs(S)
        @test !isempty(S.dat.u_train)
        @test !isempty(S.dat.u_test)
    end



    # Test for whiten_y function
    @testset "whiten_y" begin
        S = core_struct(
            prm=param_struct(
                load_name="example",
                load_path="example/example-data"), 
            dat=data_struct(),
            res=results_struct(),
            est=estimates_struct(),
            mdl=model_struct(),
            );
        S = StateSpaceAnalysis.load_data(S);
        S = StateSpaceAnalysis.whiten(S)
        @test !isempty(S.dat.y_train)
        @test !isempty(S.dat.y_test)
    end

    
    
    # Test for test_rep_ESTEP function
    @testset "test_rep_ESTEP" begin
        S = core_struct(
            prm=param_struct(
                load_name="example",
                load_path="example/example-data"), 
            dat=data_struct(),
            res=results_struct(),
            est=estimates_struct(),
            mdl=model_struct(),
            );
        S = StateSpaceAnalysis.preprocess_fit(S);
        result = test_rep_ESTEP(S)
        @test sum(result.^2) ≈ 0.0
    end



end
