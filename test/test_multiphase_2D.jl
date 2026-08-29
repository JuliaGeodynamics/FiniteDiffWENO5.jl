@testset "multiphase 2D advection" begin
    periodic2D() = ntuple(_ -> PeriodicBC(), 4)
    maxsumerr2D(p) = maximum(abs, p[1] .+ p[2] .+ p[3] .- 1)

    function regions2D(nx, ny)
        dx, dy = 1 / nx, 1 / ny
        p1 = fill(0.1, nx, ny)
        p2 = fill(0.15, nx, ny)
        for j in 1:ny, i in 1:nx
            x = (i - 0.5) * dx
            y = (j - 0.5) * dy
            if (x - 0.3)^2 + (y - 0.5)^2 < 0.12^2
                p1[i, j] = 0.75
            elseif 0.58 < x < 0.78 && 0.34 < y < 0.66
                p2[i, j] = 0.7
            end
        end
        return (p1, p2, 1 .- p1 .- p2)
    end

    function rotation_velocity2D(nx, ny)
        dx, dy = 1 / nx, 1 / ny
        vx = [-(j - 0.5) * dy + 0.5 for i in 1:(nx + 1), j in 1:ny]
        vy = [(i - 0.5) * dx - 0.5 for i in 1:nx, j in 1:(ny + 1)]
        return (; x = vx, y = vy)
    end

    @testset "staggered rotation preserves the simplex and phase integrals" begin
        nx, ny = 32, 28
        dx, dy = 1 / nx, 1 / ny
        initial = regions2D(nx, ny)
        serial = map(copy, initial)
        threaded = map(copy, initial)
        initial_integrals = map(sum, initial)
        v = rotation_velocity2D(nx, ny)
        dt = 0.12 * min(dx, dy)

        ss = MultiphaseWENOScheme(
            serial; boundary = periodic2D(), stag = true, multithreading = false
        )
        st = MultiphaseWENOScheme(
            threaded; boundary = periodic2D(), stag = true, multithreading = true
        )
        for _ in 1:100
            WENO_step!(serial, v, ss, dt, dx, dy)
            WENO_step!(threaded, v, st, dt, dx, dy)
        end

        @test maxsumerr2D(serial) <= 1024eps(Float64)
        @test all(p -> all(x -> -128eps(Float64) <= x <= 1 + 128eps(Float64), p), serial)
        for k in 1:3
            @test sum(serial[k]) ≈ initial_integrals[k] rtol = 256eps(Float64)
            @test threaded[k] == serial[k]
        end
    end

    @testset "constant composition under divergent velocity" begin
        nx, ny = 24, 20
        dx, dy = 1 / nx, 1 / ny
        constant = (0.15, 0.35, 0.5)

        staggered = ntuple(k -> fill(constant[k], nx, ny), 3)
        vx = [0.4 + 0.2 * (i - 1) * dx for i in 1:(nx + 1), j in 1:ny]
        vy = [0.3 - 0.1 * (j - 1) * dy for i in 1:nx, j in 1:(ny + 1)]
        ss = MultiphaseWENOScheme(
            staggered; boundary = periodic2D(), stag = true, multithreading = false
        )
        for _ in 1:50
            WENO_step!(staggered, (; x = vx, y = vy), ss, 0.08min(dx, dy), dx, dy)
        end
        for k in 1:3
            @test staggered[k] == fill(constant[k], nx, ny)
        end

        collocated = ntuple(k -> fill(constant[k], nx, ny), 3)
        vxc = [0.4 + 0.2 * (i - 0.5) * dx for i in 1:nx, j in 1:ny]
        vyc = [0.3 - 0.1 * (j - 0.5) * dy for i in 1:nx, j in 1:ny]
        sc = MultiphaseWENOScheme(
            collocated; boundary = periodic2D(), stag = false, multithreading = false
        )
        @test sc.vcenter === nothing
        for _ in 1:50
            WENO_step!(collocated, (; x = vxc, y = vyc), sc, 0.08min(dx, dy), dx, dy)
        end
        for k in 1:3
            @test collocated[k] == fill(constant[k], nx, ny)
        end
    end

    @testset "smooth periodic compressible material operator is fifth order" begin
        errors = Float64[]
        for n in (24, 48, 96)
            dx = 1 / n
            x = (collect(1:n) .- 0.5) .* dx
            X = repeat(x, 1, n)
            Y = repeat(x', n, 1)

            # The normal face velocities are smooth and periodic but have nonzero
            # divergence: ∂ₓvₓ + ∂ᵧvᵧ = 0.3πcos(2πx) - 0.24πsin(2πy).
            p1 = @. 0.3 + 0.07 * sinpi(2X) * cospi(2Y)
            p2 = @. 0.35 + 0.06 * cospi(2X) * sinpi(2Y)
            phases = (p1, p2, 1 .- p1 .- p2)

            xf = range(0, 1, length = n + 1)
            Xfx = repeat(xf, 1, n)
            Yfx = repeat(x', n + 1, 1)
            Xfy = repeat(x, 1, n + 1)
            Yfy = repeat(xf', n, 1)
            staggered_velocity = (
                x = 0.7 .+ 0.15 .* sinpi.(2 .* Xfx),
                y = -0.4 .+ 0.12 .* cospi.(2 .* Yfy),
            )
            vx = @. 0.7 + 0.15 * sinpi(2X)
            vy = @. -0.4 + 0.12 * cospi(2Y)

            scheme = MultiphaseWENOScheme(
                phases; boundary = periodic2D(), stag = true, multithreading = false
            )
            FiniteDiffWENO5.multiphase_WENO_flux!(phases, scheme, n, n)
            vcenter = FiniteDiffWENO5.prepare_velocity!(scheme, staggered_velocity)
            FiniteDiffWENO5.multiphase_material_semi_discretisation!(
                scheme.du, vcenter, scheme, inv(dx), inv(dx)
            )

            dp1dx = @. 0.14π * cospi(2X) * cospi(2Y)
            dp1dy = @. -0.14π * sinpi(2X) * sinpi(2Y)
            dp2dx = @. -0.12π * sinpi(2X) * sinpi(2Y)
            dp2dy = @. 0.12π * cospi(2X) * cospi(2Y)
            exact = (
                vx .* dp1dx .+ vy .* dp1dy,
                vx .* dp2dx .+ vy .* dp2dy,
                .-vx .* (dp1dx .+ dp2dx) .- vy .* (dp1dy .+ dp2dy),
            )
            push!(errors, sum(k -> sum(abs, scheme.du[k] .- exact[k]), 1:3) / (3n^2))
        end
        rates = [log2(errors[i] / errors[i + 1]) for i in 1:2]
        @test all(>(4.5), rates)
    end

    @testset "serial allocation has only fixed ENO-preparation overhead" begin
        nx, ny = 20, 18
        dx, dy = 1 / nx, 1 / ny
        v = (; x = fill(0.4, nx + 1, ny), y = fill(-0.2, nx, ny + 1))

        scalar = regions2D(nx, ny)[1]
        ws = WENOScheme(
            scalar; boundary = periodic2D(), form = :conservative, stag = true, multithreading = false
        )
        WENO_step!(scalar, v, ws, 0.1min(dx, dy), dx, dy)
        baseline = @allocated WENO_step!(scalar, v, ws, 0.1min(dx, dy), dx, dy)

        phases = regions2D(nx, ny)
        wm = MultiphaseWENOScheme(
            phases; boundary = periodic2D(), stag = true, multithreading = false
        )
        WENO_step!(phases, v, wm, 0.1min(dx, dy), dx, dy)
        # Unlike the legacy scalar staggered route, multiphase transport now prepares a
        # genuinely collocated ENO5 velocity once per step.  Its dispatch overhead is
        # fixed (not proportional to the grid or number of phases).
        @test (@allocated WENO_step!(phases, v, wm, 0.1min(dx, dy), dx, dy)) <= baseline + 64
    end
end
