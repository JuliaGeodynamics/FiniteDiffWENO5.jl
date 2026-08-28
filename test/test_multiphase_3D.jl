@testset "multiphase 3D advection" begin
    periodic3D() = ntuple(_ -> PeriodicBC(), 6)
    maxsumerr3D(p) = maximum(abs, p[1] .+ p[2] .+ p[3] .- 1)

    function smooth3D(nx, ny, nz)
        p1 = Array{Float64}(undef, nx, ny, nz)
        p2 = similar(p1)
        for k in 1:nz, j in 1:ny, i in 1:nx
            x = (i - 0.5) / nx
            y = (j - 0.5) / ny
            z = (k - 0.5) / nz
            p1[i, j, k] = 0.30 + 0.08sinpi(2x) * cospi(2y)
            p2[i, j, k] = 0.30 + 0.08sinpi(2z) * cospi(2x)
        end
        return (p1, p2, 1 .- p1 .- p2)
    end

    @testset "nonzero transport in every direction preserves invariants" begin
        nx, ny, nz = 10, 9, 8
        dx, dy, dz = 1 / nx, 1 / ny, 1 / nz
        initial = smooth3D(nx, ny, nz)
        serial = map(copy, initial)
        threaded = map(copy, initial)
        initial_integrals = map(sum, initial)
        v = (;
            x = fill(0.30, nx + 1, ny, nz),
            y = fill(-0.20, nx, ny + 1, nz),
            z = fill(0.10, nx, ny, nz + 1),
        )
        dt = 0.08min(dx, dy, dz)

        ss = MultiphaseWENOScheme(
            serial; boundary = periodic3D(), stag = true, multithreading = false)
        st = MultiphaseWENOScheme(
            threaded; boundary = periodic3D(), stag = true, multithreading = true)
        for _ in 1:30
            WENO_step!(serial, v, ss, dt, dx, dy, dz)
            WENO_step!(threaded, v, st, dt, dx, dy, dz)
        end

        @test maxsumerr3D(serial) <= 1024eps(Float64)
        @test all(p -> all(x -> -128eps(Float64) <= x <= 1 + 128eps(Float64), p), serial)
        for phase in 1:3
            @test sum(serial[phase]) ≈ initial_integrals[phase] rtol = 256eps(Float64)
            # `@threads` outlines a distinct loop kernel, so LLVM may differ by a few
            # ulps even with one Julia thread. There are no reductions or shared writes.
            @test maximum(abs, threaded[phase] .- serial[phase]) <= 4eps(Float64)
        end
    end

    @testset "constant composition under divergent staggered and collocated velocity" begin
        nx, ny, nz = 8, 9, 10
        dx, dy, dz = 1 / nx, 1 / ny, 1 / nz
        constant = (0.15, 0.35, 0.50)

        staggered = ntuple(q -> fill(constant[q], nx, ny, nz), 3)
        vx = [0.3 + 0.1 * (i - 1) * dx for i in 1:(nx + 1), j in 1:ny, k in 1:nz]
        vy = [0.2 - 0.1 * (j - 1) * dy for i in 1:nx, j in 1:(ny + 1), k in 1:nz]
        vz = [0.1 + 0.05 * (k - 1) * dz for i in 1:nx, j in 1:ny, k in 1:(nz + 1)]
        ss = MultiphaseWENOScheme(
            staggered; boundary = periodic3D(), stag = true, multithreading = false)
        for _ in 1:30
            WENO_step!(staggered, (; x = vx, y = vy, z = vz), ss,
                0.05min(dx, dy, dz), dx, dy, dz)
        end
        for q in 1:3
            @test staggered[q] == fill(constant[q], nx, ny, nz)
        end

        collocated = ntuple(q -> fill(constant[q], nx, ny, nz), 3)
        vxc = [0.3 + 0.1 * (i - 0.5) * dx for i in 1:nx, j in 1:ny, k in 1:nz]
        vyc = [0.2 - 0.1 * (j - 0.5) * dy for i in 1:nx, j in 1:ny, k in 1:nz]
        vzc = [0.1 + 0.05 * (k - 0.5) * dz for i in 1:nx, j in 1:ny, k in 1:nz]
        sc = MultiphaseWENOScheme(
            collocated; boundary = periodic3D(), stag = false, multithreading = false)
        @test sc.divv === nothing
        for _ in 1:30
            WENO_step!(collocated, (; x = vxc, y = vyc, z = vzc), sc,
                0.05min(dx, dy, dz), dx, dy, dz)
        end
        for q in 1:3
            @test collocated[q] == fill(constant[q], nx, ny, nz)
        end
    end

    @testset "serial stepping does not exceed the scalar allocation floor" begin
        n = 8
        h = 1 / n
        dt = 0.02
        v = (;
            x = fill(0.30, n + 1, n, n),
            y = fill(-0.20, n, n + 1, n),
            z = fill(0.10, n, n, n + 1),
        )

        scalar = fill(0.3, n, n, n)
        ws = WENOScheme(
            scalar; boundary = periodic3D(), stag = true, multithreading = false)
        WENO_step!(scalar, v, ws, dt, h, h, h)
        baseline = @allocated WENO_step!(scalar, v, ws, dt, h, h, h)

        phases = (fill(0.2, n, n, n), fill(0.3, n, n, n), fill(0.5, n, n, n))
        wm = MultiphaseWENOScheme(
            phases; boundary = periodic3D(), stag = true, multithreading = false)
        WENO_step!(phases, v, wm, dt, h, h, h)
        @test (@allocated WENO_step!(phases, v, wm, dt, h, h, h)) <= baseline
    end
end
