using FiniteDiffWENO5
using Chmy
using KernelAbstractions
using GLMakie

function main(; backend = CPU(), nx = 400, ny = 400)

    arch = Arch(backend)

    Lx = 1.0
    Δx = Lx / nx
    Δy = Lx / ny

    grid = UniformGrid(arch; origin = (0.0, 0.0), extent = (Lx, Lx), dims = (nx, ny))

    # Courant number
    CFL = 0.7
    period = 1

    # Grid x assumed defined
    x = range(0, length = nx, stop = Lx)
    y = range(0, length = ny, stop = Lx)
    grid_array = (x .* ones(ny)', ones(nx) .* y')

    w = π
    vx0 = -w .* (grid_array[2] .- Lx / 2)
    vy0 = w .* (grid_array[1] .- Lx / 2)


    x0 = 1 / 4
    c = 0.08

    u0 = zeros(ny, nx)

    for I in CartesianIndices((ny, nx))
        u0[I] = sign(exp(-((grid_array[1][I] - x0)^2 + (grid_array[2][I]' - x0)^2) / c^2) - 0.5) * 0.5 + 0.5
    end

    u = Field(backend, grid, Center())
    set!(u, u0)

    weno = WENOScheme(u, grid; form = :nonconservative, boundary = (2, 2, 2, 2), stag = false, multithreading = true)

    v = (
        x = Field(arch, grid, Center()),
        y = Field(arch, grid, Center()),
    )

    set!(v.x, vx0)
    set!(v.y, vy0)

    # grid size
    Δt = CFL * min(Δx, Δy)^(5 / 3)

    tmax = period / (w / (2 * π))

    t = 0
    counter = 0

    f = Figure(size = (800, 600))
    ax = Axis(f[1, 1], title = "t = $(round(t, digits = 2))")
    u_obser = Observable(u0)
    hm = heatmap!(ax, x, y, u_obser; colormap = cgrad(:roma, rev = true), colorrange = (0, 1))
    Colorbar(f[1, 2], label = "u", hm)
    display(f)

    while t < tmax
        WENO_step!(u, v, weno, Δt, Δx, Δy, grid, arch)


        t += Δt

        if t + Δt > tmax
            Δt = tmax - t
        end

        if counter % 100 == 0
            KernelAbstractions.synchronize(backend)
            u_obser[] = interior(u) |> Array
            ax.title = "t = $(round(t, digits = 2))"
        end

        counter += 1

    end

    return
end

main(backend = CPU(), nx = 400, ny = 400)
