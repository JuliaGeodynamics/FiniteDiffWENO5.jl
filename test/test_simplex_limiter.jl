using Test
using FiniteDiffWENO5

@testset "simplex_rk_stage" begin
    @testset "θ stays clamped to [0,1] when donor drifts slightly outside the simplex" begin
        # `stage` (the donor) is deliberately just outside [0,1] by roundoff, as can
        # happen when the last phase is computed as `1 .- sum(others)`. Before this
        # helper existed, the RK-stage limiter formula was hand-inlined without the
        # `clamp` that `simplex_limiter_coefficient` applies, so a barely-negative
        # donor could drive the shared θ slightly negative and corrupt every phase
        # at that cell. `simplex_rk_stage` must not reproduce that: it reuses the
        # already-clamped `limit_simplex`.
        ϵ = -1.0e-15
        initial = (0.3, 0.3, 0.4)
        stage = (ϵ, 0.4 + ϵ, 0.6 - 2ϵ)  # sums to 1 despite the negative component
        du = (10.0, -3.0, -7.0)  # pushes the already-negative phase further negative
        Δt = 0.1

        updated = FiniteDiffWENO5.simplex_rk_stage(initial, stage, du, 0.0, 1.0, Δt)

        @test all(isfinite, updated)
        @test all(x -> -1.0e-12 <= x <= 1 + 1.0e-12, updated)
        @test sum(updated) ≈ 1.0 rtol = 0 atol = 1.0e-12
    end

    @testset "reduces to the identity when the forward-Euler step stays admissible" begin
        initial = (0.2, 0.3, 0.5)
        stage = (0.25, 0.35, 0.4)
        du = (0.1, -0.05, -0.05)  # candidate = stage - Δt*du stays in [0,1]
        Δt = 0.1
        candidate = (0.25 - 0.01, 0.35 + 0.005, 0.4 + 0.005)

        updated = FiniteDiffWENO5.simplex_rk_stage(initial, stage, du, 0.0, 1.0, Δt)

        @test updated[1] ≈ candidate[1] rtol = 0 atol = 1.0e-14
        @test updated[2] ≈ candidate[2] rtol = 0 atol = 1.0e-14
        @test updated[3] ≈ candidate[3] rtol = 0 atol = 1.0e-14
    end

    @testset "matches an explicit SSP convex combination for a≠0" begin
        initial = (0.2, 0.3, 0.5)
        stage = (0.1, 0.1, 0.8)
        du = (5.0, 0.0, -5.0)  # forces the limiter to engage on phase 1
        Δt = 0.1
        a, b = 0.75, 0.25

        updated = FiniteDiffWENO5.simplex_rk_stage(initial, stage, du, a, b, Δt)
        limited = FiniteDiffWENO5.limit_simplex(
            (stage[1] - Δt * du[1], stage[2] - Δt * du[2], stage[3] - Δt * du[3]), stage,
        )
        expected = ntuple(k -> a * initial[k] + b * limited[k], 3)

        for k in 1:3
            @test updated[k] ≈ expected[k] rtol = 0 atol = 1.0e-14
        end
        @test sum(updated) ≈ a * sum(initial) + b rtol = 0 atol = 1.0e-14
    end
end
