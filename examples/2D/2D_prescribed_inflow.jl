using FiniteDiffWENO5
using GLMakie

"""
Advect a tangentially varying prescribed-inflow profile through a 2D domain.

`west_temperature` has one value per cell along the west face, ordered from
bottom to top.  Because the x-velocity is positive, that face is inflow and
the profile is selected by the upwind flux.  Reversing the x-velocity makes
the west face outflow, so the prescribed values are then ignored.
"""
function main(; nx = 240, ny = 120, tmax = 0.8)
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx / nx, Ly / ny
    x = range(dx / 2, Lx - dx / 2; length = nx)
    y = range(dy / 2, Ly - dy / 2; length = ny)

    temperature = fill(300.0, nx, ny)
    west_temperature = @. 300.0 + 250.0 * exp(-((y - Ly / 2) / 0.16)^2)

    velocity = (
        x = ones(nx + 1, ny),
        y = zeros(nx, ny + 1),
    )
    boundary = AdvectionBC(
        west = PrescribedInflowBC(west_temperature),
        east = ExtrapolateBC(),
        bot = ExtrapolateBC(),
        top = ExtrapolateBC(),
    )
    # Non-conservative (material) transport: the conservative path's inflow closure
    # is only first-order accurate at the boundary by construction (see
    # src/WENO5/conservative_flux.jl), which would visibly blur the whole point of
    # this demo — a smooth prescribed profile advecting in from the west face.
    # Non-conservative substitutes the prescribed state directly, at full order.
    weno = WENOScheme(
        temperature;
        form = :nonconservative,
        boundary,
        stag = true,
        multithreading = true,
    )

    cfl = 0.5
    dt = cfl * min(dx, dy)
    t = 0.0
    step = 0

    figure = Figure(size = (960, 520))
    temperature_observable = Observable(copy(temperature))
    ax_temperature = Axis(
        figure[1, 1],
        xlabel = "x",
        ylabel = "y",
        title = "Prescribed west inflow, t = 0.00",
    )
    heatmap = heatmap!(
        ax_temperature,
        x,
        y,
        temperature_observable;
        colormap = :thermal,
        colorrange = (300.0, maximum(west_temperature)),
    )
    Colorbar(figure[1, 2], heatmap, label = "temperature")

    ax_profile = Axis(
        figure[1, 3],
        xlabel = "temperature",
        ylabel = "y",
        title = "West-face profile",
    )
    lines!(ax_profile, west_temperature, y, color = :black, linewidth = 3)
    xlims!(ax_profile, 280.0, maximum(west_temperature) + 20.0)

    display(figure)
    while t < tmax
        dt_step = min(dt, tmax - t)
        WENO_step!(temperature, velocity, weno, dt_step, dx, dy)
        t += dt_step
        step += 1

        if step % 4 == 0 || t == tmax
            temperature_observable[] = copy(temperature)
            ax_temperature.title = "Prescribed west inflow, t = $(round(t; digits = 2))"
            sleep(0.1)
        end
    end

    return temperature, west_temperature
end

main()
