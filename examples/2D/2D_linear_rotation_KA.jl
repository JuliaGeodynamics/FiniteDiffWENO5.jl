using FiniteDiffWENO5
using KernelAbstractions
using GLMakie

function main(; backend = CPU(), nx = 400, ny = 400)

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

    w = π
    vx0 = -w .* (grid_array[2] .- Lx / 2)
    vy0 = w .* (grid_array[1] .- Lx / 2)

    x0 = 1 / 4
    c = 0.08

    u0 = zeros(ny, nx)

    for I in CartesianIndices((ny, nx))
        u0[I] = sign(exp(-((grid_array[1][I] - x0)^2 + (grid_array[2][I]' - x0)^2) / c^2) - 0.5) * 0.5 + 0.5
    end

    u = KernelAbstractions.zeros(backend, Float64, nx, ny)
    copyto!(u, u0)

    weno = WENOScheme(u, backend; boundary = (2, 2, 2, 2), stag = false)


    v = (;
        x = KernelAbstractions.zeros(backend, Float64, nx, ny),
        y = KernelAbstractions.zeros(backend, Float64, nx, ny),
    )

    copyto!(v.x, vx0)
    copyto!(v.y, vy0)

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
        WENO_step!(u, v, weno, Δt, Δx, Δy, backend)


        t += Δt

        if t + Δt > tmax
            Δt = tmax - t
        end

        if counter % 100 == 0
            KernelAbstractions.synchronize(backend)
            u_obser[] = u |> Array
            ax.title = "t = $(round(t, digits = 2))"
            display(f)
        end

        counter += 1

    end

    return nothing
end

main(backend = CPU(), nx = 400, ny = 400)
