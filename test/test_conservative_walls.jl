using Test
using FiniteDiffWENO5
using KernelAbstractions

# Face velocities that vanish exactly on the walls of [0,1]^N (sinpi is exact
# at integers), with ∇·v ≠ 0 so the conservative form matters.
wall_grid(n) = (((1:n) .- 0.5) ./ n, (0:n) ./ n)

function wall_case_2D(n)
    x, xf = wall_grid(n)
    u0 = [1 + 0.5cospi(2xi) * sinpi(yj) for xi in x, yj in x]
    v = (;
        x = [sinpi(xi) * (1 + 0.5yj) for xi in xf, yj in x],
        y = [0.5sinpi(yj) * (1 + xi) for xi in x, yj in xf],
    )
    return u0, v
end

relerr(u, u0) = abs(sum(u) - sum(u0)) / sum(u0)

@testset "conservative walls" begin
    scheme(u) = WENOScheme(u; form = :conservative, stag = true, multithreading = false)

    @testset "1D: Σu conserved with zero wall velocity" begin
        n = 64
        x, xf = wall_grid(n)
        u0 = 1 .+ 0.5 .* cospi.(2 .* x)
        u = copy(u0)
        weno = scheme(u)
        for _ in 1:10
            WENO_step!(u, (; x = sinpi.(xf)), weno, 0.4 / n, 1 / n)
        end
        @test relerr(u, u0) <= 1.0e-13
    end

    @testset "2D: Σu conserved with zero wall velocity" begin
        n = 32
        u0, v = wall_case_2D(n)
        u = copy(u0)
        weno = scheme(u)
        for _ in 1:10
            WENO_step!(u, v, weno, 0.2 / n, 1 / n, 1 / n)
        end
        @test relerr(u, u0) <= 1.0e-13
    end

    @testset "3D: Σu conserved with zero wall velocity" begin
        n = 16
        x, xf = wall_grid(n)
        u0 = [1 + 0.3cospi(2xi) * cospi(2yj) * sinpi(zk) for xi in x, yj in x, zk in x]
        v = (;
            x = [sinpi(xi) for xi in xf, _ in x, _ in x],
            y = [sinpi(yj) for _ in x, yj in xf, _ in x],
            z = [sinpi(zk) for _ in x, _ in x, zk in xf],
        )
        u = copy(u0)
        weno = scheme(u)
        for _ in 1:10
            WENO_step!(u, v, weno, 0.1 / n, 1 / n, 1 / n, 1 / n)
        end
        @test relerr(u, u0) <= 1.0e-13
    end

    @testset "2D tuple step: every field conserved" begin
        n = 32
        u0, v = wall_case_2D(n)
        a, b = copy(u0), 2 .* u0
        weno = scheme(a)
        for _ in 1:10
            WENO_step!((a, b), v, weno, 0.2 / n, 1 / n, 1 / n; u_min = (0.0, 0.0), u_max = (0.0, 0.0))
        end
        @test relerr(a, u0) <= 1.0e-13
        @test relerr(b, 2 .* u0) <= 1.0e-13
    end

    @testset "2D tuple step on the KernelAbstractions CPU backend" begin
        n = 32
        u0, v = wall_case_2D(n)
        a, b = copy(u0), 2 .* u0
        weno = WENOScheme(a, CPU(); form = :conservative, stag = true)
        for _ in 1:10
            WENO_step!((a, b), v, weno, 0.2 / n, 1 / n, 1 / n, CPU(); u_min = (0.0, 0.0), u_max = (0.0, 0.0))
        end
        @test relerr(a, u0) <= 1.0e-13
        @test relerr(b, 2 .* u0) <= 1.0e-13
    end

    @testset "nonzero wall velocity gives the upwind flux" begin
        n = 16
        u = rand(n)
        vel = (; x = fill(1.0, n + 1))
        weno = scheme(u)
        FiniteDiffWENO5.conservative_semi_discretisation_weno5!(
            weno.du, u, FiniteDiffWENO5.prepare_velocity!(weno, vel), weno, n, Float64(n);
            vface = vel,
        )
        @test weno.fl.x[1] == u[1] && weno.fr.x[1] == 0
        @test weno.fl.x[n + 1] == u[n] && weno.fr.x[n + 1] == 0
    end

    @testset "face velocity selection" begin
        n = 16
        u = ones(n)
        vel = (; x = ones(n + 1))
        weno_stag = scheme(u)
        FiniteDiffWENO5.prepare_velocity!(weno_stag, vel)
        weno_colloc = WENOScheme(u; form = :conservative, stag = false, multithreading = false)
        weno_material = WENOScheme(u; form = :nonconservative, stag = true, multithreading = false)
        @test FiniteDiffWENO5.conservative_face_velocity(weno_stag, vel, nothing) === vel
        @test FiniteDiffWENO5.conservative_face_velocity(weno_stag, weno_stag.vcenter, nothing) === nothing
        @test FiniteDiffWENO5.conservative_face_velocity(weno_stag, weno_stag.vcenter, vel) === vel
        # Tuple steps forward their input as `given`; a prepared vcenter is no face data.
        @test FiniteDiffWENO5.conservative_face_velocity(weno_stag, weno_stag.vcenter, weno_stag.vcenter) === nothing
        @test FiniteDiffWENO5.conservative_face_velocity(weno_colloc, (; x = ones(n)), nothing) === nothing
        @test FiniteDiffWENO5.conservative_face_velocity(weno_material, vel, nothing) === nothing
    end
end
