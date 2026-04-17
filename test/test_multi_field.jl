@testset "Multi-field advection tests" begin

    @testset "1D multi-field equivalence" begin
        nx = 200
        x_min = -1.0
        x_max = 1.0
        Lx = x_max - x_min
        x = range(x_min, stop = x_max, length = nx)

        CFL = 0.4
        period = 1

        # Two different initial conditions
        u1_init = @. exp(-((x - 0.0)^2) / 0.01)
        u2_init = @. exp(-((x + 0.3)^2) / 0.02)
        u3_init = @. exp(-((x - 0.4)^2) / 0.005)

        # Single-field references
        u1_single = copy(u1_init)
        u2_single = copy(u2_init)
        u3_single = copy(u3_init)

        # Multi-field tuple
        u1_multi = copy(u1_init)
        u2_multi = copy(u2_init)
        u3_multi = copy(u3_init)

        weno = WENOScheme(u1_init; boundary = (2, 2), stag = true)

        a = (; x = ones(nx + 1))
        Δx = x[2] - x[1]
        Δt = CFL * Δx^(5 / 3)
        tmax = period * (Lx + Δx) / maximum(abs.(a.x))

        t = 0.0
        while t < tmax
            # Single-field calls
            WENO_step!(u1_single, a, weno, Δt, Δx; u_min = 0.0, u_max = 1.0)
            WENO_step!(u2_single, a, weno, Δt, Δx; u_min = 0.0, u_max = 1.0)
            WENO_step!(u3_single, a, weno, Δt, Δx; u_min = -1.0, u_max = 2.0)

            # Multi-field call
            WENO_step!((u1_multi, u2_multi, u3_multi), a, weno, Δt, Δx;
                u_min = (0.0, 0.0, -1.0), u_max = (1.0, 1.0, 2.0))

            t += Δt
            if t + Δt > tmax
                Δt = tmax - t
            end
        end

        @test u1_multi ≈ u1_single
        @test u2_multi ≈ u2_single
        @test u3_multi ≈ u3_single
    end


    @testset "2D multi-field equivalence" begin
        nx = 50
        ny = 50
        Lx = 1.0
        Δx = Lx / nx
        Δy = Lx / ny

        x = range(0, length = nx, stop = Lx)
        y = range(0, length = ny, stop = Lx)
        grid_array = (x .* ones(ny)', ones(nx) .* y')

        CFL = 0.7
        period = 1

        x0 = 1 / 4
        c1 = 0.08
        c2 = 0.12

        u1_init = zeros(ny, nx)
        u2_init = zeros(ny, nx)
        for I in CartesianIndices((ny, nx))
            u1_init[I] = exp(-((grid_array[1][I] - x0)^2 + (grid_array[2][I]' - x0)^2) / c1^2)
            u2_init[I] = exp(-((grid_array[1][I] - 0.5)^2 + (grid_array[2][I]' - 0.5)^2) / c2^2)
        end

        vx0 = ones(nx, ny)
        vy0 = ones(nx, ny)
        v = (; x = vx0, y = vy0)

        # Single-field references
        u1_single = copy(u1_init)
        u2_single = copy(u2_init)

        # Multi-field
        u1_multi = copy(u1_init)
        u2_multi = copy(u2_init)

        weno = WENOScheme(u1_init; boundary = (2, 2, 2, 2), stag = false, multithreading = false)

        Δt = CFL * min(Δx, Δy)^(5 / 3)
        tmax = period * Lx / max(maximum(abs.(vx0)), maximum(abs.(vy0)))

        t = 0.0
        while t < tmax
            WENO_step!(u1_single, v, weno, Δt, Δx, Δy; u_min = 0.0, u_max = 1.0)
            WENO_step!(u2_single, v, weno, Δt, Δx, Δy; u_min = 0.0, u_max = 1.0)

            WENO_step!((u1_multi, u2_multi), v, weno, Δt, Δx, Δy;
                u_min = (0.0, 0.0), u_max = (1.0, 1.0))

            t += Δt
            if t + Δt > tmax
                Δt = tmax - t
            end
        end

        @test u1_multi ≈ u1_single
        @test u2_multi ≈ u2_single
    end


    @testset "3D multi-field equivalence" begin
        nx = 20
        ny = 20
        nz = 20

        L = 1.0
        Δx = L / nx
        Δy = L / ny
        Δz = L / nz

        x = range(0, stop = L, length = nx)
        y = range(0, stop = L, length = ny)
        z = range(0, stop = L, length = nz)

        X = reshape(x, 1, 1, nx) .* ones(ny, nz, 1)
        Y = reshape(y, 1, ny, 1) .* ones(nx, 1, nz)
        Z = reshape(z, nz, 1, 1) .* ones(1, ny, nx)

        CFL = 0.7
        period = 1

        u1_init = zeros(ny, nx, nz)
        u2_init = zeros(ny, nx, nz)
        for I in CartesianIndices((ny, nx, nz))
            u1_init[I] = exp(-((X[I] - 0.25)^2 + (Y[I] - 0.25)^2 + (Z[I] - 0.5)^2) / 0.01)
            u2_init[I] = exp(-((X[I] - 0.5)^2 + (Y[I] - 0.5)^2 + (Z[I] - 0.5)^2) / 0.02)
        end

        vx0 = ones(size(X))
        vy0 = ones(size(Y))
        vz0 = zeros(size(Z))
        v = (; x = vx0, y = vy0, z = vz0)

        u1_single = copy(u1_init)
        u2_single = copy(u2_init)
        u1_multi = copy(u1_init)
        u2_multi = copy(u2_init)

        weno = WENOScheme(u1_init; boundary = (2, 2, 2, 2, 2, 2), stag = false, multithreading = false)

        Δt = CFL * min(Δx, Δy, Δz)^(5 / 3)
        tmax = period * L / max(maximum(abs.(vx0)), maximum(abs.(vy0)))

        t = 0.0
        while t < tmax
            WENO_step!(u1_single, v, weno, Δt, Δx, Δy, Δz; u_min = 0.0, u_max = 1.0)
            WENO_step!(u2_single, v, weno, Δt, Δx, Δy, Δz; u_min = 0.0, u_max = 1.0)

            WENO_step!((u1_multi, u2_multi), v, weno, Δt, Δx, Δy, Δz;
                u_min = (0.0, 0.0), u_max = (1.0, 1.0))

            t += Δt
            if t + Δt > tmax
                Δt = tmax - t
            end
        end

        @test u1_multi ≈ u1_single
        @test u2_multi ≈ u2_single
    end


    @testset "1D multi-field KA CPU equivalence" begin
        backend = CPU()
        nx = 200
        x_min = -1.0
        x_max = 1.0
        Lx = x_max - x_min
        x = range(x_min, stop = x_max, length = nx)

        CFL = 0.4
        period = 1

        u1_init = KernelAbstractions.zeros(backend, Float64, nx)
        u2_init = KernelAbstractions.zeros(backend, Float64, nx)
        for i in 1:nx
            u1_init[i] = exp(-((x[i] - 0.0)^2) / 0.01)
            u2_init[i] = exp(-((x[i] + 0.3)^2) / 0.02)
        end

        u1_single = copy(u1_init)
        u2_single = copy(u2_init)
        u1_multi = copy(u1_init)
        u2_multi = copy(u2_init)

        weno = WENOScheme(u1_init, backend; boundary = (2, 2), stag = true)

        a = (; x = KernelAbstractions.ones(backend, Float64, nx + 1))
        Δx = x[2] - x[1]
        Δt = CFL * Δx^(5 / 3)
        tmax = period * (Lx + Δx) / 1.0

        t = 0.0
        while t < tmax
            WENO_step!(u1_single, a, weno, Δt, Δx, backend; u_min = 0.0, u_max = 1.0)
            WENO_step!(u2_single, a, weno, Δt, Δx, backend; u_min = 0.0, u_max = 1.0)

            WENO_step!((u1_multi, u2_multi), a, weno, Δt, Δx, backend;
                u_min = (0.0, 0.0), u_max = (1.0, 1.0))

            t += Δt
            if t + Δt > tmax
                Δt = tmax - t
            end
        end

        @test u1_multi ≈ u1_single
        @test u2_multi ≈ u2_single
    end
end
