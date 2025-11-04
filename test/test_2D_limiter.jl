@testset "2D limiter tests" begin
    @testset "2D linear case" begin

        # Number of grid points
        nx = 100
        ny = 100
        Lx = 1.0
        Δx = Lx / nx
        Δy = Lx / ny

        x = range(0, stop = Lx, length = nx)

        # Courant number
        CFL = 0.7
        period = 1

        # Grid x assumed defined
        x = range(0, length = nx, stop = Lx)
        y = range(0, length = ny, stop = Lx)
        grid_array = (x .* ones(ny)', ones(nx) .* y')

        vx0 = ones(nx, ny)
        vy0 = ones(nx, ny)

        v = (; x = vy0, y = vx0)

        x0 = 1 / 4
        c = 0.08

        u0 = zeros(ny, nx)

        for I in CartesianIndices((ny, nx))
            u0[I] = sign(exp(-((grid_array[1][I] - x0)^2 + (grid_array[2][I]' - x0)^2) / c^2) - 0.5) * 0.5 + 0.5
        end

        u = copy(u0)
        weno = WENOScheme(u; boundary = (2, 2, 2, 2), stag = false, lim_ZS = true, multithreading = true)


        # grid size
        Δt = CFL * min(Δx, Δy)^(5 / 3)

        tmax = period * Lx / max(maximum(abs.(vx0)), maximum(abs.(vy0)))

        t = 0

        while t < tmax
            WENO_step!(u, v, weno, Δt, Δx, Δy; u_min = 0.0, u_max = 1.0)

            t += Δt

            if t + Δt > tmax
                Δt = tmax - t
            end
        end

        @test isapprox(sum(u), sum(u0); atol = 1.0e-6)
        @test isapprox(maximum(u), maximum(u0); atol = 1.0e-2)
    end


    @testset "2D linear case KA CPU" begin
        backend = CPU()
        nx = 100
        ny = 100
        stag = true

        Lx = 1.0
        Δx = Lx / nx
        Δy = Lx / ny

        # Courant number
        CFL = 0.7
        period = 1

        # Grid x assumed defined
        x = range(0, length = nx, stop = Lx)
        y = range(0, length = ny, stop = Lx)
        grid_array = (x .* ones(ny)', ones(nx) .* y')

        if stag
            vx0 = ones(nx + 1, ny)
            vy0 = ones(nx, ny + 1)
        else
            vx0 = ones(nx, ny)
            vy0 = ones(nx, ny)
        end

        v = (; x = vy0, y = vx0)

        x0 = 1 / 4
        c = 0.08

        u0 = zeros(ny, nx)

        for I in CartesianIndices((ny, nx))
            u0[I] = sign(exp(-((grid_array[1][I] - x0)^2 + (grid_array[2][I]' - x0)^2) / c^2) - 0.5) * 0.5 + 0.5
        end

        u = KernelAbstractions.zeros(backend, Float64, nx, ny)
        copyto!(u, u0)

        weno = WENOScheme(u, backend; boundary = (2, 2, 2, 2), stag = stag, limiter = true, multithreading = true)

        if stag
            v = (;
                x = KernelAbstractions.zeros(backend, Float64, nx + 1, ny),
                y = KernelAbstractions.zeros(backend, Float64, nx, ny + 1),
            )
        else
            v = (;
                x = KernelAbstractions.zeros(backend, Float64, nx, ny),
                y = KernelAbstractions.zeros(backend, Float64, nx, ny),
            )
        end

        copyto!(v.x, vx0)
        copyto!(v.y, vy0)

        # grid size
        Δt = CFL * min(Δx, Δy)^(5 / 3)

        tmax = period * Lx / max(maximum(abs.(vx0)), maximum(abs.(vy0)))

        t = 0
        counter = 0

        mass_ini = sum(u0) * Δx * Δy

        while t < tmax
            WENO_step!(u, v, weno, Δt, Δx, Δy, backend; u_min = 0.0, u_max = 1.0)


            t += Δt

            if t + Δt > tmax
                Δt = tmax - t
            end

            counter += 1

        end
        @test isapprox(sum(u), sum(u0); atol = 1.0e-6)
        @test isapprox(maximum(u), maximum(u0); atol = 1.0e-2)
    end
end
