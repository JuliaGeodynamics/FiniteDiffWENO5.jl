import FiniteDiffWENO5: combined_weno_betas, weno_betas,
    multiphase_reconstruction_upwind, multiphase_reconstruction_downwind,
    weno5_reconstruction_upwind, weno5_reconstruction_downwind,
    simplex_limiter_coefficient, limit_simplex, nphases

# three five-point stencils whose values sum to one at every stencil position
const STENCILS_3 = (
    (0.1, 0.15, 0.3, 0.55, 0.4),
    (0.25, 0.35, 0.2, 0.1, 0.35),
    (0.65, 0.5, 0.5, 0.35, 0.25),
)

@testset "multiphase reconstruction" begin

    @testset "combined smoothness indicators" begin
        χ = (13 / 12, 1 / 4)

        β_reference = ntuple(3) do r
            sum(weno_betas(STENCILS_3[k]..., χ)[r] for k in eachindex(STENCILS_3)) /
                length(STENCILS_3)
        end
        @test all(combined_weno_betas(STENCILS_3, χ) .≈ β_reference)

        # A single phase reduces to the scalar indicators. This is numerical rather than
        # bitwise agreement: the `+ zero` accumulation and the `* inv(NP)` rescaling let
        # the compiler contract the `@muladd` expressions in `weno_betas` differently, so
        # the two paths can differ by one ULP.
        @test all(
            isapprox.(
                combined_weno_betas((STENCILS_3[1],), χ),
                weno_betas(STENCILS_3[1]..., χ);
                rtol = 8eps(Float64),
            )
        )

        # the mean is symmetric under phase permutation
        for perm in ((1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1))
            permuted = ntuple(i -> STENCILS_3[perm[i]], 3)
            @test all(combined_weno_betas(permuted, χ) .≈ combined_weno_betas(STENCILS_3, χ))
        end
    end

    @testset "shared weights reconstruct to a partition of unity" begin
        γ = (0.1, 0.6, 0.3)
        χ = (13 / 12, 1 / 4)
        ζ = (1 / 3, 7 / 6, 11 / 6, 1 / 6, 5 / 6)
        ϵ = eps(Float64)

        up = multiphase_reconstruction_upwind(STENCILS_3, χ, γ, ζ, ϵ)
        dn = multiphase_reconstruction_downwind(STENCILS_3, χ, γ, ζ, ϵ)

        @test length(up) == 3
        @test sum(up) ≈ 1.0 atol = 64eps(Float64)
        @test sum(dn) ≈ 1.0 atol = 64eps(Float64)

        # permuting the phases permutes the reconstruction and leaves the sum intact
        perm = (3, 1, 2)
        permuted = ntuple(i -> STENCILS_3[perm[i]], 3)
        up_perm = multiphase_reconstruction_upwind(permuted, χ, γ, ζ, ϵ)
        for i in 1:3
            @test up_perm[i] ≈ up[perm[i]]
        end

        # a single phase reproduces the scalar reconstruction; the weight normalisation
        # divides out the ULP-level difference in the shared indicators
        @test multiphase_reconstruction_upwind((STENCILS_3[1],), χ, γ, ζ, ϵ)[1] ≈
            weno5_reconstruction_upwind(STENCILS_3[1]..., χ, γ, ζ, ϵ) rtol = 8eps(Float64)
        @test multiphase_reconstruction_downwind((STENCILS_3[1],), χ, γ, ζ, ϵ)[1] ≈
            weno5_reconstruction_downwind(STENCILS_3[1]..., χ, γ, ζ, ϵ) rtol = 8eps(Float64)

        # independent per-phase weights do NOT preserve the sum: this is the defect the
        # shared-weight design exists to remove
        independent = ntuple(k -> weno5_reconstruction_upwind(STENCILS_3[k]..., χ, γ, ζ, ϵ), 3)
        @test abs(sum(independent) - 1) > 1.0e-3
    end

    @testset "simplex limiter" begin
        donor = (0.1, 0.25, 0.65)

        # lower-bound overshoot
        high = (-0.04, 0.31, 0.73)
        θ = simplex_limiter_coefficient(high, donor)
        limited = limit_simplex(high, donor, θ)
        @test 0.0 <= θ < 1.0
        @test all(x -> 0.0 <= x <= 1.0, limited)
        @test sum(limited) ≈ 1.0 atol = 16eps(Float64)
        @test all(k -> limited[k] == donor[k] + θ * (high[k] - donor[k]), eachindex(limited))
        @test limit_simplex(high, donor) == limited

        # upper-bound overshoot
        donor_hi = (0.8, 0.1, 0.1)
        high_hi = (1.05, 0.02, -0.07)
        θ_hi = simplex_limiter_coefficient(high_hi, donor_hi)
        limited_hi = limit_simplex(high_hi, donor_hi, θ_hi)
        @test all(x -> 0.0 <= x <= 1.0, limited_hi)
        @test sum(limited_hi) ≈ 1.0 atol = 16eps(Float64)

        # already admissible: θ == 1 and the state is returned untouched
        admissible = (0.12, 0.24, 0.64)
        @test simplex_limiter_coefficient(admissible, donor) == 1.0
        @test limit_simplex(admissible, donor, 1.0) == admissible

        # pure donor state
        @test simplex_limiter_coefficient(donor, donor) == 1.0
        @test limit_simplex(donor, donor) == donor

        # a donor sitting exactly on a bound forces full reversion
        donor_edge = (0.0, 0.4, 0.6)
        @test simplex_limiter_coefficient((-0.1, 0.45, 0.65), donor_edge) == 0.0

        # all six permutations give permuted results and identical θ
        for perm in ((1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1))
            d = ntuple(i -> donor[perm[i]], 3)
            h = ntuple(i -> high[perm[i]], 3)
            @test simplex_limiter_coefficient(h, d) == θ
            @test limit_simplex(h, d, θ) == ntuple(i -> limited[perm[i]], 3)
        end

        # Float32
        donor32 = Float32.(donor)
        high32 = Float32.(high)
        θ32 = simplex_limiter_coefficient(high32, donor32)
        limited32 = limit_simplex(high32, donor32, θ32)
        @test θ32 isa Float32
        @test all(x -> 0.0f0 <= x <= 1.0f0, limited32)
        @test sum(limited32) ≈ 1.0f0 atol = 64eps(Float32)

        # two phases and five phases
        @test sum(limit_simplex((-0.2, 1.2), (0.3, 0.7))) ≈ 1.0 atol = 16eps(Float64)
        d5 = (0.2, 0.2, 0.2, 0.2, 0.2)
        h5 = (-0.1, 0.3, 0.25, 0.3, 0.25)
        @test all(x -> 0.0 <= x <= 1.0, limit_simplex(h5, d5))
        @test sum(limit_simplex(h5, d5)) ≈ 1.0 atol = 16eps(Float64)
    end

    @testset "allocation and inference" begin
        γ = (0.1, 0.6, 0.3)
        χ = (13 / 12, 1 / 4)
        ζ = (1 / 3, 7 / 6, 11 / 6, 1 / 6, 5 / 6)
        ϵ = eps(Float64)
        donor = (0.1, 0.25, 0.65)
        high = (-0.04, 0.31, 0.73)

        @test (@inferred combined_weno_betas(STENCILS_3, χ)) isa NTuple{3, Float64}
        @test (@inferred multiphase_reconstruction_upwind(STENCILS_3, χ, γ, ζ, ϵ)) isa
            NTuple{3, Float64}
        @test (@inferred simplex_limiter_coefficient(high, donor)) isa Float64
        @test (@inferred limit_simplex(high, donor)) isa NTuple{3, Float64}

        combined_weno_betas(STENCILS_3, χ)
        multiphase_reconstruction_upwind(STENCILS_3, χ, γ, ζ, ϵ)
        multiphase_reconstruction_downwind(STENCILS_3, χ, γ, ζ, ϵ)
        limit_simplex(high, donor)
        # Julia 1.10 on Linux may materialize one 32-byte isbits tuple at these
        # call boundaries. Bound that compiler-version overhead tightly so a
        # genuine heap allocation still fails the regression test.
        @test (@allocated combined_weno_betas(STENCILS_3, χ)) <= 32
        @test (@allocated multiphase_reconstruction_upwind(STENCILS_3, χ, γ, ζ, ϵ)) <= 32
        @test (@allocated multiphase_reconstruction_downwind(STENCILS_3, χ, γ, ζ, ϵ)) <= 32
        @test (@allocated limit_simplex(high, donor)) <= 32
    end
end
