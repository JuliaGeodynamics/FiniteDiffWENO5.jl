using Test
using FiniteDiffWENO5

@testset "local Lax-Friedrichs splitting" begin
    @testset "stencil_lf_speed" begin
        @test FiniteDiffWENO5.stencil_lf_speed(0.0, -3.0, 1.0, 2.0, 0.5, -0.1) == 3.0
    end

    @testset "stagnant rows are untouched" begin
        n = 32
        Δ = inv(n)
        u0 = [i <= 16 ? 1.0 : 0.0 for i in 1:n, j in 1:n]
        v = (; x = [j > 16 ? 1.0 : 0.0 for i in 1:n, j in 1:n], y = zeros(n, n))
        boundary = (PeriodicBC(), PeriodicBC(), ExtrapolateBC(), ExtrapolateBC())
        u = copy(u0)
        weno = WENOScheme(u; form = :conservative, boundary, stag = false, multithreading = false)
        WENO_step!(u, v, weno, 0.4Δ, Δ, Δ)
        @test u[:, 1:16] == u0[:, 1:16]       # f± ≡ 0 in rows without velocity
        @test u[:, 17:end] != u0[:, 17:end]   # the moving rows' step moved
    end

    # Bound: twice the overshoot of upwind (nonconservative) WENO on this setup (4.5e-6).
    @testset "no new extrema under solid-body rotation" begin
        tol_rot = 1.0e-5
        n = 64
        Δ = inv(n)
        x = ((1:n) .- 0.5) .* Δ
        u = [(0.35 <= xi <= 0.65 && 0.35 <= yj <= 0.65) ? 1.0 : 0.0 for xi in x, yj in x]
        v = (; x = [-(yj - 0.5) for xi in x, yj in x], y = [xi - 0.5 for xi in x, yj in x])
        weno = WENOScheme(u; form = :conservative, boundary = ntuple(_ -> ExtrapolateBC(), 4), stag = false, multithreading = false)
        for _ in 1:40
            WENO_step!(u, v, weno, 0.4Δ / 0.5, Δ, Δ)
        end
        @test minimum(u) >= -tol_rot
        @test maximum(u) <= 1 + tol_rot
    end
end
