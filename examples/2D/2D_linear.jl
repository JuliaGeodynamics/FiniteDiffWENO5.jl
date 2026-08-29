using FiniteDiffWENO5
using GLMakie

function main(; nx = 400, ny = 400)

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
    grid = (x .* ones(ny)', ones(nx) .* y')

    vx0 = ones(nx, ny)
    vy0 = ones(nx, ny)

    v = (; x = vy0, y = vx0)

    x0 = 1 / 4
    c = 0.08

    u0 = zeros(ny, nx)

    for I in CartesianIndices((ny, nx))
        u0[I] = sign(exp(-((grid[1][I] - x0)^2 + (grid[2][I]' - x0)^2) / c^2) - 0.5) * 0.5 + 0.5
    end

    u = copy(u0)
    weno = WENOScheme(u; form = :nonconservative, boundary = (2, 2, 2, 2), stag = false, multithreading = true)


    # grid size
    Δt = CFL * min(Δx, Δy)^(5 / 3)

    tmax = period * Lx / max(maximum(abs.(vx0)), maximum(abs.(vy0)))

    t = 0
    counter = 0

    f = Figure(size = (800, 600))
    ax = Axis(f[1, 1], title = "t = $(round(t, digits = 2))")
    u_obser = Observable(u0)
    hm = heatmap!(ax, x, y, u_obser; colormap = cgrad(:roma, rev = true), colorrange = (0, 1))
    Colorbar(f[1, 2], label = "u", hm)
    display(f)

    while t < tmax
        WENO_step!(u, v, weno, Δt, Δx, Δy)


        t += Δt

        if t + Δt > tmax
            Δt = tmax - t
        end

        if counter % 100 == 0
            u_obser[] = u
            ax.title = "t = $(round(t, digits = 2))"
        end

        counter += 1

    end

    return
end

main(nx = 400, ny = 400)
