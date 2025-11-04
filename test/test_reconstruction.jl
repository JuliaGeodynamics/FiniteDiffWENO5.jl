@testset "reconstruction tests" begin
    @testset "Linear case" begin
        u = [1.0, 2.0, 3.0, 4.0, 5.0]
        weno = WENOScheme(u)

        @unpack χ, γ, ζ, ϵ = weno

        f_up = FiniteDiffWENO5.weno5_reconstruction_upwind(u[1], u[2], u[3], u[4], u[5], χ, γ, ζ, ϵ)
        f_down = FiniteDiffWENO5.weno5_reconstruction_downwind(u[1], u[2], u[3], u[4], u[5], χ, γ, ζ, ϵ)

        @test f_up ≈ 3.5
        @test f_down ≈ 2.5
    end

    @testset "Zhang-Shu limiter basics" begin
        ϵθ = 1e-20
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

    @testset "Randomized limiter stress test" begin
        ϵθ = 1e-20
        u_min, u_max = 0.0, 1.0
        for _ in 1:100
            u_avg = rand()
            u_val = rand() * 2 .- 0.5  # random in [-0.5, 1.5]
            val = FiniteDiffWENO5.zhang_shu_limit(u_val, u_avg, u_min, u_max, ϵθ)
            @test u_min - 1e-10 ≤ val ≤ u_max + 1e-10
        end
    end

end


