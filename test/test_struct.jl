@testset "struct tests" begin
    @testset "1D" begin
        u0 = ones(3)
        weno = WENOScheme(u0)
        @test weno.γ == (0.1, 0.6, 0.3)
        @test weno.χ == (13 / 12, 1 / 4)
        @test weno.ζ == (1 / 3, 7 / 6, 11 / 6, 1 / 6, 5 / 6)
        @test weno.ϵ == eps(Float64)
        @test weno.boundary == (ExtrapolateBC(), ExtrapolateBC())
        @test all(weno.fl.x .== 0.0)
        @test all(weno.fr.x .== 0.0)
        @test all(weno.du .== 0.0)
        @test all(weno.ut .== 0.0)

        # test type of input
        u0 = [3.0f0]
        weno = WENOScheme(u0)
        @test typeof(weno.γ) == NTuple{3, Float32}
        @test eps(Float32) == weno.ϵ
        @test all(weno.fl.x .== 0.0f0)
    end
    @testset "2D" begin
        u0 = ones(3, 3)
        weno = WENOScheme(u0)
        @test weno.boundary == ntuple(i -> ExtrapolateBC(), 4)
        @test size(weno.fl.x) == (4, 3)
        @test size(weno.fr.x) == (4, 3)
        @test size(weno.fl.y) == (3, 4)
        @test size(weno.fr.y) == (3, 4)
    end
    @testset "3D" begin
        u0 = ones(3, 3, 3)
        weno = WENOScheme(u0)
        @test weno.boundary == ntuple(i -> ExtrapolateBC(), 6)
        @test size(weno.fl.x) == (4, 3, 3)
        @test size(weno.fr.x) == (4, 3, 3)
        @test size(weno.fl.y) == (3, 4, 3)
        @test size(weno.fr.y) == (3, 4, 3)
        @test size(weno.fl.z) == (3, 3, 4)
        @test size(weno.fr.z) == (3, 3, 4)
    end
end
