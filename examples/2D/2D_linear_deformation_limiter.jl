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

    # Staggered grids:
    # vx at x-faces → (nx+1, ny)
    x_vx = range(0, stop = Lx, length = nx + 1)
    y_vx = range(Δy / 2, stop = Lx - Δy / 2, length = ny)

    # vy at y-faces → (nx, ny+1)
    x_vy = range(Δx / 2, stop = Lx - Δx / 2, length = nx)
    y_vy = range(0, stop = Lx, length = ny + 1)

    # Make 2D coordinate arrays with correct orientation
    X_vx = repeat(x_vx, 1, ny)         # (nx+1, ny)
    Y_vx = repeat(y_vx', nx + 1, 1)      # (nx+1, ny)

    X_vy = repeat(x_vy, 1, ny + 1)       # (nx, ny+1)
    Y_vy = repeat(y_vy', nx, 1)        # (nx, ny+1)

    # Define velocity field (example: divergence-free vortex)
    vx0 = .-2π .* sin.(π .* X_vx) .* cos.(π .* Y_vx)  # (nx+1, ny)
    vy0 = 2π .* cos.(π .* X_vy) .* sin.(π .* Y_vy)   # (nx, ny+1)

    v = (; x = vx0, y = vy0)

    x0 = 1 / 4
    c = 0.08

    u0 = zeros(ny, nx)

    for I in CartesianIndices((ny, nx))
        u0[I] = sign(exp(-((grid[1][I] - x0)^2 + (grid[2][I]' - x0)^2) / c^2) - 0.5) * 0.5 + 0.5
    end

    u = copy(u0)
    weno = WENOScheme(u; boundary = (2, 2, 2, 2), stag = true, lim_ZS = true, multithreading = true)


    # grid size
    Δt = CFL * min(Δx, Δy)^(5 / 3)

    tmax = 1
    t = 0
    counter = 0

    f = Figure(size = (800, 600))
    ax = Axis(f[1, 1], title = "t = $(round(t, digits = 2))")
    u_obser = Observable(u0)
    hm = heatmap!(ax, x, y, u_obser; colormap = cgrad(:roma, rev = true), colorrange = (0, 1))
    Colorbar(f[1, 2], label = "u", hm)
    display(f)
    counter_half = 0

    while t < tmax
        WENO_step!(u, v, weno, Δt, Δx, Δy; u_min = 0.0, u_max = 1.0)


        if t > tmax / 2 && counter_half == 0
            println("reversing velocity field...")
            vx0 .= .- vx0
            vy0 .= .- vy0
            counter_half += 1
        end

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
