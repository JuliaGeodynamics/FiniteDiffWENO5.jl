using FiniteDiffWENO5
using GLMakie

function main(; nx = 50, ny = 50, nz = 50)
    L = 1.0
    Δx = L / nx
    Δy = L / ny
    Δz = L / nz

    x = range(0, stop = L, length = nx)
    y = range(0, stop = L, length = ny)
    z = range(0, stop = L, length = nz)

    # Courant number
    CFL = 0.7
    period = 1

    # 3D grid
    X = reshape(x, 1, nx, 1) .* ones(ny, 1, nz)
    Y = reshape(y, ny, 1, 1) .* ones(1, nx, nz)
    Z = reshape(z, 1, 1, nz) .* ones(ny, nx, 1)

    vx0 = ones(size(X))
    vy0 = ones(size(Y))
    vz0 = zeros(size(X)) # Rotation in XY plane only

    v = (; x = vx0, y = vy0, z = vz0)

    x0 = 1 / 4
    c = 0.08

    u0 = zeros(ny, nx, nz)
    for I in CartesianIndices((ny, nx, nz))
        u0[I] = exp(-((X[I] - x0)^2 + (Y[I] - x0)^2 + (Z[I] - 0.5)^2) / c^2)
    end

    u = copy(u0)
    weno = WENOScheme(u; form = :nonconservative, boundary = (2, 2, 2, 2, 2, 2), stag = false, multithreading = true)

    Δt = CFL * min(Δx, Δy, Δz)^(5 / 3)
    tmax = period * L / max(maximum(abs.(vx0)), maximum(abs.(vy0)), maximum(abs.(vz0)))
    t = 0
    counter = 0

    f = Figure(size = (800, 600))
    ax = Axis(f[1, 1], title = "t = $(round(t, digits = 2))")
    u_obser = Observable(u[:, :, div(nz, 2)])
    heatmap!(ax, u_obser; colormap = cgrad(:roma, rev = true), colorrange = (0, 1.0))
    Colorbar(f[1, 2], label = "u")
    display(f)

    while t < tmax
        WENO_step!(u, v, weno, Δt, Δx, Δy, Δz)

        t += Δt
        if t + Δt > tmax
            Δt = tmax - t
        end

        if counter % 50 == 0
            u_obser[] = u[:, :, div(nz, 2)]
            ax.title = "t = $(round(t, digits = 2))"
        end

        counter += 1
    end
    return
end

main()
