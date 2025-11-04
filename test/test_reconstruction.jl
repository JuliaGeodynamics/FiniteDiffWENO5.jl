@testset "reconstruction tests" begin

    # --- Constant field: reconstruction should return the same value ---
    @testset "Constant field" begin
        u = fill(3.14, 5)
        weno = WENOScheme(u)
        @unpack χ, γ, ζ, ϵ = weno

        f_up = FiniteDiffWENO5.weno5_reconstruction_upwind(u..., χ, γ, ζ, ϵ)
        f_down = FiniteDiffWENO5.weno5_reconstruction_downwind(u..., χ, γ, ζ, ϵ)

        @test f_up ≈ 3.14 atol = 1.0e-12
        @test f_down ≈ 3.14 atol = 1.0e-12
    end

    # --- Linear field: should reconstruct 5th-order accurate interface ---
    @testset "Linear field" begin
        u = [1.0, 2.0, 3.0, 4.0, 5.0]
        weno = WENOScheme(u)
        @unpack χ, γ, ζ, ϵ = weno

        f_up = FiniteDiffWENO5.weno5_reconstruction_upwind(u..., χ, γ, ζ, ϵ)
        f_down = FiniteDiffWENO5.weno5_reconstruction_downwind(u..., χ, γ, ζ, ϵ)

        @test f_up ≈ 3.5 atol = 1.0e-12
        @test f_down ≈ 2.5 atol = 1.0e-12
    end

    # --- Quadratic field: should reproduce polynomial exactly up to 5th order ---
    @testset "Quadratic field" begin
        x = -2:2
        u = float.(x .^ 2)  # smooth convex profile
        weno = WENOScheme(float(u))
        @unpack χ, γ, ζ, ϵ = weno

        fl = (2u[1] - 13u[2] + 47u[3] + 27u[4] - 3u[5]) / 60.0
        fr = (-3u[1] + 27u[2] + 47u[3] - 13u[4] + 2u[5]) / 60.0

        f_up = FiniteDiffWENO5.weno5_reconstruction_upwind(u..., χ, γ, ζ, ϵ)
        f_down = FiniteDiffWENO5.weno5_reconstruction_downwind(u..., χ, γ, ζ, ϵ)

        @test f_up ≈ fl atol = 1.0e-3
        @test f_down ≈ fr atol = 1.0e-3
    end

    # --- Discontinuous field: should remain non-oscillatory ---
    @testset "Discontinuous field" begin
        u = [1.0, 1.0, 1.0, 10.0, 10.0]
        weno = WENOScheme(u)
        @unpack χ, γ, ζ, ϵ = weno

        f_up = FiniteDiffWENO5.weno5_reconstruction_upwind(u..., χ, γ, ζ, ϵ)
        f_down = FiniteDiffWENO5.weno5_reconstruction_downwind(u..., χ, γ, ζ, ϵ)

        # Should not overshoot — values must remain within range
        @test 1.0 ≤ f_up + eps(f_down) ≤ 10.0
        @test 1.0 ≤ f_down + eps(f_down) ≤ 10.0
    end

    # --- Zhang-Shu limiter tests ---
    @testset "Zhang-Shu limiter basics" begin
        ϵθ = 1.0e-20
        u_min, u_max = 0.0, 1.0

        # Case 1: inside range → no change
        val = FiniteDiffWENO5.zhang_shu_limit(0.5, 0.5, u_min, u_max, ϵθ)
        @test val ≈ 0.5

        # Case 2: above u_max → should limit to ~u_max
        val = FiniteDiffWENO5.zhang_shu_limit(1.5, 0.5, u_min, u_max, ϵθ)
        @test u_min ≤ val ≤ u_max

        # Case 3: below u_min → should limit to ~u_min
        val = FiniteDiffWENO5.zhang_shu_limit(-0.5, 0.5, u_min, u_max, ϵθ)
        @test u_min ≤ val ≤ u_max

        # Case 4: exact u_val = u_avg (no diff)
        @test FiniteDiffWENO5.zhang_shu_limit(0.5, 0.5, u_min, u_max, ϵθ) == 0.5
    end

    # --- Randomized stress test for limiter ---
    @testset "Randomized limiter stress test" begin
        ϵθ = 1.0e-20
        u_min, u_max = 0.0, 1.0
        for _ in 1:100
            u_avg = rand()
            u_val = rand() * 2 .- 0.5  # random in [-0.5, 1.5]
            val = FiniteDiffWENO5.zhang_shu_limit(u_val, u_avg, u_min, u_max, ϵθ)
            @test u_min - 1.0e-10 ≤ val ≤ u_max + 1.0e-10
        end
    end
end
