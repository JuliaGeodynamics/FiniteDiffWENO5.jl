using FiniteDiffWENO5
using Chmy
using KernelAbstractions
using GLMakie

function main(; backend = CPU(), nx = 400, ny = 400, stag = true)

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

    u = Field(backend, grid, Center())
    set!(u, u0)

    weno = WENOScheme(u, grid; form = :nonconservative, boundary = (1, 1, 1, 1), stag = stag, multithreading = true)

    if stag
        v = VectorField(backend, grid)
    else
        v = (
            x = Field(arch, grid, Center()),
            y = Field(arch, grid, Center()),
        )
    end

    set!(v.x, vy0)
    set!(v.y, vx0)

    # grid size
    Δt = CFL * min(Δx, Δy)^(5 / 3)

    tmax = period * Lx / max(maximum(abs.(vx0)), maximum(abs.(vy0)))

    t = 0
    counter = 0

    mass_ini = sum(u0) * Δx * Δy

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

            mass_ratio = (sum(u) * Δx * Δy) / mass_ini

            u_obser[] = interior(u) |> Array
            ax.title = "t = $(round(t, digits = 2)), mass ratio = $(round(mass_ratio, digits = 6))"
        end

        counter += 1

    end

    return
end


main(backend = CPU(), nx = 400, ny = 400, stag = true)
