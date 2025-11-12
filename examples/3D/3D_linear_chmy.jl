using FiniteDiffWENO5
using Chmy
using KernelAbstractions
using CairoMakie

function main(; backend = CPU(), nx = 50, ny = 50, nz = 50)

    arch = Arch(backend)

    Lx = 1.0
    Δx = Lx / nx
    Δy = Lx / ny
    Δz = Lx / nz

    grid = UniformGrid(arch; origin = (0.0, 0.0, 0.0), extent = (Lx, Lx, Lx), dims = (nx, ny, nz))

    # Courant number
    CFL = 0.7
    period = 1

    # 3D grid
    x = range(0, length = nx, stop = Lx)
    y = range(0, length = ny, stop = Lx)
    z = range(0, length = nz, stop = Lx)

    X = reshape(x, 1, nx, 1) .* ones(ny, 1, nz)
    Y = reshape(y, ny, 1, 1) .* ones(1, nx, nz)
    Z = reshape(z, 1, 1, nz) .* ones(ny, nx, 1)

    X3D = X .+ 0 .* Y .+ 0 .* Z
    Y3D = 0 .* X .+ Y .+ 0 .* Z
    Z3D = 0 .* X .+ 0 .* Y .+ Z

    vx0 = ones(size(X3D))
    vy0 = ones(size(Y3D))
    vz0 = zeros(size(Z3D)) # Rotation in XY plane only

    v = (;
        x = Field(arch, grid, Center()),
        y = Field(arch, grid, Center()),
        z = Field(arch, grid, Center()),
    )
    set!(v.x, vy0)
    set!(v.y, vx0)
    set!(v.z, vz0)

    x0 = 1 / 4
    c = 0.08

    u0 = zeros(ny, nx, nz)
    for I in CartesianIndices((ny, nx, nz))
        u0[I] = exp(-((X3D[I] - x0)^2 + (Y3D[I] - x0)^2 + (Z3D[I] - 0.5)^2) / c^2)
    end

    u = Field(backend, grid, Center())
    set!(u, u0)
    weno = WENOScheme(u, grid; boundary = (2, 2, 2, 2, 2, 2), stag = false)

    Δt = CFL * min(Δx, Δy, Δz)^(5 / 3)
    tmax = period * Lx / max(maximum(abs.(vx0)), maximum(abs.(vy0)), maximum(abs.(vz0)))
    t = 0
    counter = 0

    f = Figure(size = (800, 600))
    ax = Axis(f[1, 1], title = "t = $(round(t, digits = 2))")
    u_obser = Observable(u0[:, :, div(nz, 2)])
    hm = heatmap!(ax, u_obser; colormap = cgrad(:roma, rev = true), colorrange = (0, 1))
    Colorbar(f[1, 2], label = "u", hm)
    display(f)

    while t < tmax
        WENO_step!(u, v, weno, Δt, Δx, Δy, Δz, grid, arch)

        t += Δt
        if t + Δt > tmax
            Δt = tmax - t
        end


        if counter % 10 == 0
            if backend == CPU()
                KernelAbstractions.synchronize(backend)
                u_obser[] = (interior(u) |> Array)[:, :, div(nz, 2)]
                ax.title = "t = $(round(t, digits = 2))"
                display(f)
            end
        end

        counter += 1
    end

    KernelAbstractions.synchronize(backend)
    u_obser[] = (interior(u) |> Array)[:, :, div(nz, 2)]
    ax.title = "t = $(round(t, digits = 2))"

    return save("weno5_cuda.png", f)
end

# using CUDA
# main(backend=CUDABackend())
main()
