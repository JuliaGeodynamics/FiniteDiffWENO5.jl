@testset "multiphase 1D advection" begin

    # piecewise-constant three-phase composition summing to one everywhere
    function fronts(nx)
        dx = 1 / nx
        x = range(dx / 2, 1 - dx / 2, length = nx)
        p1 = [xi < 0.3 ? 0.8 : 0.1 for xi in x]
        p2 = [0.3 < xi < 0.6 ? 0.7 : 0.15 for xi in x]
        return (p1, p2, 1 .- p1 .- p2)
    end

    # smooth field held strictly inside the simplex so the limiter stays inactive and the
    # formal order of the reconstruction is what is measured
    function smooth(nx)
        dx = 1 / nx
        x = range(dx / 2, 1 - dx / 2, length = nx)
        p1 = @. 0.35 + 0.1 * sinpi(2x)
        p2 = @. 0.35 + 0.1 * cospi(2x)
        return (p1, p2, 1 .- p1 .- p2)
    end

    periodic1D() = (PeriodicBC(), PeriodicBC())
    maxsumerr(p) = maximum(abs, p[1] .+ p[2] .+ p[3] .- 1)

    @testset "sum error against the sequential route" begin
        nx = 200
        dx = 1 / nx
        dt = 0.4dx
        v = (; x = fill(1.0, nx + 1))
        nsteps = 250

        # negative control: independent per-phase weights, same grid, velocity, dt, state
        seq = fronts(nx)
        wscalar = WENOScheme(seq[1]; boundary = periodic1D(), stag = true, multithreading = false)
        for _ in 1:nsteps
            WENO_step!(seq, v, wscalar, dt, dx; u_min = (0.0, 0.0, 0.0), u_max = (1.0, 1.0, 1.0))
        end
        sequential_error = maxsumerr(seq)

        # the simultaneous operator on the identical problem
        sim = fronts(nx)
        scheme = MultiphaseWENOScheme(sim; boundary = periodic1D(), stag = true, multithreading = false)
        for _ in 1:nsteps
            WENO_step!(sim, v, scheme, dt, dx)
        end
        simultaneous_error = maxsumerr(sim)

        # measured 2.6e-8 for the sequential route, 3.4e-14 (153 eps, saturating) here
        @test sequential_error > 1.0e-10
        @test simultaneous_error < 1024eps(Float64)
        @test simultaneous_error < sequential_error / 1.0e5

        # bounds are respected
        @test all(p -> all(x -> -64eps(Float64) <= x <= 1 + 64eps(Float64), p), sim)
    end

    @testset "single step stays within 64eps" begin
        nx = 200
        dx = 1 / nx
        p = fronts(nx)
        v = (; x = fill(1.0, nx + 1))
        scheme = MultiphaseWENOScheme(p; boundary = periodic1D(), stag = true, multithreading = false)
        WENO_step!(p, v, scheme, 0.4dx, dx)
        @test maxsumerr(p) <= 64eps(Float64)
    end

    @testset "conservation under divergence-free periodic transport" begin
        nx = 200
        dx = 1 / nx
        p = fronts(nx)
        initial = map(sum, p)
        v = (; x = fill(1.0, nx + 1))
        scheme = MultiphaseWENOScheme(p; boundary = periodic1D(), stag = true, multithreading = false)
        for _ in 1:250
            WENO_step!(p, v, scheme, 0.4dx, dx)
        end
        for k in 1:3
            @test sum(p[k]) ≈ initial[k] rtol = 256eps(Float64)
        end
    end

    @testset "constant composition survives a divergent staggered velocity" begin
        # the flux divergence and the ϕ∇·v source must cancel discretely
        nx = 64
        dx = 1 / nx
        p = (fill(0.2, nx), fill(0.3, nx), fill(0.5, nx))
        v = (; x = collect(range(0.5, 1.5, length = nx + 1)))
        scheme = MultiphaseWENOScheme(p; boundary = periodic1D(), stag = true, multithreading = false)
        for _ in 1:5
            WENO_step!(p, v, scheme, 0.2dx, dx)
        end
        # measured exact: the cancellation is bitwise, not merely within tolerance
        @test maximum(abs, p[1] .- 0.2) == 0.0
        @test maximum(abs, p[2] .- 0.3) == 0.0
        @test maximum(abs, p[3] .- 0.5) == 0.0
    end

    @testset "collocated velocity needs no divergence source" begin
        nx = 64
        dx = 1 / nx
        p = (fill(0.2, nx), fill(0.3, nx), fill(0.5, nx))
        v = (; x = collect(range(0.5, 1.5, length = nx)))
        scheme = MultiphaseWENOScheme(p; boundary = periodic1D(), stag = false, multithreading = false)
        @test scheme.divv === nothing
        for _ in 1:5
            WENO_step!(p, v, scheme, 0.2dx, dx)
        end
        @test maximum(abs, p[1] .- 0.2) == 0.0
        @test maximum(abs, p[3] .- 0.5) == 0.0
    end

    @testset "smooth transport converges at fifth order" begin
        errors = Float64[]
        for nx in (40, 80, 160)
            dx = 1 / nx
            exact = smooth(nx)
            p = map(copy, exact)
            # every phase stays clear of 0 and 1, so the limiter never engages
            @test all(q -> all(x -> 0.1 <= x <= 0.6, q), exact)
            v = (; x = fill(1.0, nx + 1))
            scheme = MultiphaseWENOScheme(p; boundary = periodic1D(), stag = true, multithreading = false)
            # dt ∝ dx^(5/3) makes the 3rd-order SSP-RK3 error fifth order in space
            nt = ceil(Int, 1 / (0.4 * dx^(5 / 3)))
            dt = 1 / nt
            for _ in 1:nt
                WENO_step!(p, v, scheme, dt, dx)
            end
            push!(errors, sum(k -> sum(abs, p[k] .- exact[k]), 1:3) / (3nx))
        end
        rates = [log2(errors[i] / errors[i + 1]) for i in 1:(length(errors) - 1)]
        @test all(>(4.0), rates)
        @test errors[end] < errors[1]
    end

    @testset "phase order is a relabelling" begin
        nx = 128
        dx = 1 / nx
        v = (; x = fill(1.0, nx + 1))
        perm = (3, 1, 2)

        base = fronts(nx)
        s1 = MultiphaseWENOScheme(base; boundary = periodic1D(), stag = true, multithreading = false)
        for _ in 1:20
            WENO_step!(base, v, s1, 0.4dx, dx)
        end

        original = fronts(nx)
        permuted = ntuple(i -> original[perm[i]], 3)
        s2 = MultiphaseWENOScheme(permuted; boundary = periodic1D(), stag = true, multithreading = false)
        for _ in 1:20
            WENO_step!(permuted, v, s2, 0.4dx, dx)
        end

        # Equivariance is numerical, not bitwise: the shared smoothness indicators are a
        # sum over phases, and floating-point addition is not associative, so relabelling
        # changes the summation order and the last bit with it.
        for i in 1:3
            @test permuted[i] ≈ base[perm[i]] rtol = 64eps(Float64)
            @test maximum(abs, permuted[i] .- base[perm[i]]) < 1.0e-14
        end
    end

    @testset "prescribed inflow acts only at inflow" begin
        nx = 60
        dx = 1 / nx
        inflow = PrescribedInflowBC((0.7, 0.2, 0.1))
        v = (; x = fill(1.0, nx + 1))   # flow to the right: west is inflow, east is outflow

        with_bc = (fill(0.2, nx), fill(0.3, nx), fill(0.5, nx))
        s = MultiphaseWENOScheme(
            with_bc; boundary = (inflow, ExtrapolateBC()), stag = true, multithreading = false
        )
        for _ in 1:20
            WENO_step!(with_bc, v, s, 0.4dx, dx)
        end

        without_bc = (fill(0.2, nx), fill(0.3, nx), fill(0.5, nx))
        s0 = MultiphaseWENOScheme(
            without_bc; boundary = (ExtrapolateBC(), ExtrapolateBC()), stag = true, multithreading = false
        )
        for _ in 1:20
            WENO_step!(without_bc, v, s0, 0.4dx, dx)
        end

        # the prescribed composition must actually reach the domain
        @test maximum(abs, with_bc[1] .- without_bc[1]) > 1.0e-8
        @test with_bc[1][1] > without_bc[1][1]
        # and it must not disturb the far downstream end
        @test with_bc[1][end] ≈ without_bc[1][end] atol = 1.0e-12
        # the simplex still holds
        @test maxsumerr(with_bc) < 1024eps(Float64)

        # reversing the flow makes the same face an outflow, so nothing is imposed
        rev = (fill(0.2, nx), fill(0.3, nx), fill(0.5, nx))
        vrev = (; x = fill(-1.0, nx + 1))
        srev = MultiphaseWENOScheme(
            rev; boundary = (inflow, ExtrapolateBC()), stag = true, multithreading = false
        )
        for _ in 1:20
            WENO_step!(rev, vrev, srev, 0.4dx, dx)
        end
        @test maximum(abs, rev[1] .- 0.2) < 1.0e-12
    end

    @testset "phase count must match the scheme" begin
        p3 = (zeros(8), zeros(8), zeros(8))
        scheme = MultiphaseWENOScheme(p3; boundary = periodic1D(), stag = true)
        v = (; x = fill(1.0, 9))
        @test_throws DimensionMismatch WENO_step!((zeros(8), zeros(8)), v, scheme, 0.1, 0.1)

        # The simplex bounds are not caller-supplied. Passing them must fail loudly. In
        # particular the tuple form must NOT fall through to the sequential
        # `WENO_step!(u::Tuple, args...; u_min::Tuple, u_max::Tuple)`, which would advect
        # each phase independently and silently destroy the sum invariant.
        @test_throws MethodError WENO_step!(
            p3, v, scheme, 0.1, 0.1; u_min = (0.0, 0.0, 0.0), u_max = (1.0, 1.0, 1.0)
        )
        @test_throws TypeError WENO_step!(p3, v, scheme, 0.1, 0.1; u_min = 0.0, u_max = 1.0)
    end

    @testset "serial stepping allocates no more than the scalar operator" begin
        # The scalar `WENO_step!` itself allocates a small fixed amount per call on this
        # package (16 bytes, from the `@maybe_threads` ternary), so an absolute zero is
        # not the right target. Measure the scalar baseline in the same process and
        # require the multiphase operator not to exceed it: that is self-calibrating and
        # cannot rot across Julia versions.
        nx = 64
        dx = 1 / nx

        u = fronts(nx)[1]
        v = (; x = fill(1.0, nx + 1))
        wscalar = WENOScheme(u; boundary = periodic1D(), stag = true, multithreading = false)
        WENO_step!(u, v, wscalar, 0.4dx, dx)
        baseline = @allocated WENO_step!(u, v, wscalar, 0.4dx, dx)

        p = fronts(nx)
        scheme = MultiphaseWENOScheme(p; boundary = periodic1D(), stag = true, multithreading = false)
        WENO_step!(p, v, scheme, 0.4dx, dx)          # warm up
        @test (@allocated WENO_step!(p, v, scheme, 0.4dx, dx)) <= baseline

        pc = fronts(nx)
        vc = (; x = fill(1.0, nx))
        sc = MultiphaseWENOScheme(pc; boundary = periodic1D(), stag = false, multithreading = false)
        WENO_step!(pc, vc, sc, 0.4dx, dx)
        @test (@allocated WENO_step!(pc, vc, sc, 0.4dx, dx)) <= baseline

        # crucially, allocation must not scale with the phase count or the grid
        p5 = ntuple(_ -> fill(0.2, nx), 5)
        s5 = MultiphaseWENOScheme(p5; boundary = periodic1D(), stag = true, multithreading = false)
        WENO_step!(p5, v, s5, 0.4dx, dx)
        @test (@allocated WENO_step!(p5, v, s5, 0.4dx, dx)) <= baseline

        big = fronts(4nx)
        vbig = (; x = fill(1.0, 4nx + 1))
        sbig = MultiphaseWENOScheme(big; boundary = periodic1D(), stag = true, multithreading = false)
        WENO_step!(big, vbig, sbig, 0.1dx, dx)
        @test (@allocated WENO_step!(big, vbig, sbig, 0.1dx, dx)) <= baseline
    end
end
