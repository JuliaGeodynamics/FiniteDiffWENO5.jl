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

@testset "multiphase struct tests" begin
    import FiniteDiffWENO5: nphases

    @testset "1D" begin
        phases = (zeros(6), zeros(6), zeros(6))
        scheme = MultiphaseWENOScheme(phases)
        @test scheme.γ == (0.1, 0.6, 0.3)
        @test scheme.χ == (13 / 12, 1 / 4)
        @test scheme.ζ == (1 / 3, 7 / 6, 11 / 6, 1 / 6, 5 / 6)
        @test scheme.ϵ == eps(Float64)
        @test scheme.boundary == (ExtrapolateBC(), ExtrapolateBC())
        @test nphases(scheme) == 3

        # per-phase buffers are concrete NTuples, not abstract Tuples
        @test scheme.du isa NTuple{3, Vector{Float64}}
        @test scheme.ut isa NTuple{3, Vector{Float64}}
        @test scheme.fl.x isa NTuple{3, Vector{Float64}}
        @test scheme.fr.x isa NTuple{3, Vector{Float64}}
        @test all(length(f) == 7 for f in scheme.fl.x)
        @test all(all(iszero, f) for f in scheme.fl.x)
        @test all(all(iszero, d) for d in scheme.du)

        # divv exists only on the staggered path
        @test scheme.divv === nothing
        staggered = MultiphaseWENOScheme(phases; stag = true)
        @test staggered.divv isa Vector{Float64}
        @test size(staggered.divv) == (6,)

        # The constructed object is fully concrete, which is what the hot loop needs.
        # The constructor's own return type is a two-way Union over `divv` whenever `stag`
        # is a runtime value, because `divv` is a `Vector` when staggered and `nothing`
        # when collocated. That Union is resolved once, outside any loop.
        @test isconcretetype(typeof(scheme))
        @test isconcretetype(typeof(MultiphaseWENOScheme(phases; stag = true)))
        @test (@inferred nphases(scheme)) == 3
        @test Base.infer_return_type(MultiphaseWENOScheme, Tuple{typeof(phases)}) !== Any

        # element type is taken from the phases
        phases32 = (zeros(Float32, 6), zeros(Float32, 6))
        scheme32 = MultiphaseWENOScheme(phases32)
        @test scheme32.ϵ == eps(Float32)
        @test typeof(scheme32.γ) == NTuple{3, Float32}
        @test scheme32.du isa NTuple{2, Vector{Float32}}
        @test nphases(scheme32) == 2

        # no scalar-only options are exposed
        @test !hasproperty(scheme, :lim_ZS)
        @test !hasproperty(scheme, :upwind_mode)
        @test_throws MethodError MultiphaseWENOScheme(phases; lim_ZS = true)
        @test_throws MethodError MultiphaseWENOScheme(phases; upwind_mode = true)
    end

    @testset "2D and 3D shapes" begin
        phases2 = (zeros(3, 4), zeros(3, 4), zeros(3, 4))
        s2 = MultiphaseWENOScheme(phases2; stag = true)
        @test s2.boundary == ntuple(i -> ExtrapolateBC(), 4)
        @test all(size(f) == (4, 4) for f in s2.fl.x)
        @test all(size(f) == (3, 5) for f in s2.fl.y)
        @test all(size(f) == (4, 4) for f in s2.fr.x)
        @test all(size(f) == (3, 5) for f in s2.fr.y)
        @test size(s2.divv) == (3, 4)

        phases3 = (zeros(3, 3, 3), zeros(3, 3, 3))
        s3 = MultiphaseWENOScheme(phases3)
        @test s3.boundary == ntuple(i -> ExtrapolateBC(), 6)
        @test all(size(f) == (4, 3, 3) for f in s3.fl.x)
        @test all(size(f) == (3, 4, 3) for f in s3.fl.y)
        @test all(size(f) == (3, 3, 4) for f in s3.fl.z)
    end

    @testset "construction errors" begin
        # fewer than two phases
        @test_throws ArgumentError MultiphaseWENOScheme((zeros(4),))
        # not arrays
        @test_throws ArgumentError MultiphaseWENOScheme((1.0, 2.0))
        # mismatched element type
        @test_throws ArgumentError MultiphaseWENOScheme((zeros(4), zeros(Float32, 4)))
        # mismatched axes
        @test_throws DimensionMismatch MultiphaseWENOScheme((zeros(4), zeros(5)))
        # mismatched dimensionality
        @test_throws DimensionMismatch MultiphaseWENOScheme((zeros(4), zeros(2, 2)))
        # mismatched concrete array family
        @test_throws ArgumentError MultiphaseWENOScheme((zeros(4), view(zeros(8), 1:4)))
        # only the implemented spatial dimensions are accepted
        @test_throws ArgumentError MultiphaseWENOScheme((fill(0.4), fill(0.6)))
        @test_throws ArgumentError MultiphaseWENOScheme((zeros(2, 2, 2, 2), zeros(2, 2, 2, 2)))
    end
end
