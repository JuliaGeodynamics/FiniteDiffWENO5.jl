using FiniteDiffWENO5
using GLMakie

function main(; nx = 400, ny = 400)

    Lx = 1.0
    Δx = Lx / nx
    Δy = Lx / ny

    # Courant number
    CFL = 0.7
    period = 1

    # Grid
    x = range(0, length = nx, stop = Lx)
    y = range(0, length = ny, stop = Lx)
    grid = (x .* ones(ny)', ones(nx) .* y')

    # Shared velocity field
    vx0 = ones(nx, ny)
    vy0 = ones(nx, ny)
    v = (; x = vy0, y = vx0)

    # Three chemical components with different initial conditions
    x0 = 1 / 4
    c_width1 = 0.08
    c_width2 = 0.06
    c_width3 = 0.10

    c1 = zeros(ny, nx)
    c2 = zeros(ny, nx)
    c3 = zeros(ny, nx)

    for I in CartesianIndices((ny, nx))
        c1[I] = sign(exp(-((grid[1][I] - x0)^2 + (grid[2][I]' - x0)^2) / c_width1^2) - 0.5) * 0.5 + 0.5
        c2[I] = exp(-((grid[1][I] - 0.5)^2 + (grid[2][I]' - 0.5)^2) / c_width2^2)
        c3[I] = exp(-((grid[1][I] - 0.75)^2 + (grid[2][I]' - 0.25)^2) / c_width3^2)
    end

    c1_0 = copy(c1)
    c2_0 = copy(c2)
    c3_0 = copy(c3)

    # Create a single WENOScheme shared by all fields
    weno = WENOScheme(c1; boundary = (2, 2, 2, 2), stag = false, multithreading = true)

    Δt = CFL * min(Δx, Δy)^(5 / 3)
    tmax = period * Lx / max(maximum(abs.(vx0)), maximum(abs.(vy0)))

    t = 0
    counter = 0

    # Plot setup
    f = Figure(size = (1200, 400))
    ax1 = Axis(f[1, 1], title = "Component 1")
    ax2 = Axis(f[1, 2], title = "Component 2")
    ax3 = Axis(f[1, 3], title = "Component 3")
    c1_obs = Observable(c1_0)
    c2_obs = Observable(c2_0)
    c3_obs = Observable(c3_0)
    heatmap!(ax1, x, y, c1_obs; colormap = cgrad(:roma, rev = true), colorrange = (0, 1))
    heatmap!(ax2, x, y, c2_obs; colormap = cgrad(:roma, rev = true), colorrange = (0, 1))
    heatmap!(ax3, x, y, c3_obs; colormap = cgrad(:roma, rev = true), colorrange = (0, 1))
    display(f)

    while t < tmax
        # Advect all 3 components in a single call
        WENO_step!((c1, c2, c3), v, weno, Δt, Δx, Δy;
            u_min = (0.0, 0.0, 0.0),
            u_max = (1.0, 1.0, 1.0))

        t += Δt

        if t + Δt > tmax
            Δt = tmax - t
        end

        if counter % 100 == 0
            c1_obs[] = c1
            c2_obs[] = c2
            c3_obs[] = c3
        end

        counter += 1
    end

    return nothing
end

main(nx = 400, ny = 400)
