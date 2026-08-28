using FiniteDiffWENO5
using GLMakie

nx, ny = 96, 96
Lx, Ly = 1.0, 1.0
Δx, Δy = Lx / nx, Ly / ny
x = (collect(1:nx) .- 0.5) .* Δx
y = (collect(1:ny) .- 0.5) .* Δy

# A smooth three-material composition. Normalising positive marker functions gives
# pointwise fractions in [0, 1] whose sum is one.
w1 = [0.05 + exp(-((xi - 0.32)^2 + (yj - 0.5)^2) / 0.015) for xi in x, yj in y]
w2 = [0.05 + exp(-((xi - 0.68)^2 + (yj - 0.5)^2) / 0.015) for xi in x, yj in y]
w3 = fill(0.15, nx, ny)
total = w1 .+ w2 .+ w3
phases = (w1 ./ total, w2 ./ total, w3 ./ total)

# Divergence-free solid-body rotation on a staggered grid.
x_faces = range(0.0, Lx; length = nx + 1)
y_faces = range(0.0, Ly; length = ny + 1)
velocity = (
    x = [-(yj - Ly / 2) for _ in x_faces, yj in y],
    y = [(xi - Lx / 2) for xi in x, _ in y_faces],
)
boundary = ntuple(_ -> PeriodicBC(), 4)
scheme = MultiphaseWENOScheme(
    phases; boundary = boundary, stag = true, multithreading = true
)

initial_integrals = map(sum, phases)
Δt = 0.25 * min(Δx, Δy)^(5 / 3)
rotation_period = 2π
t = 0.0
step = 0

sum_title(t, integrals) = "Sum of phases, t = $(round(t, digits = 3))\n" *
    "∫ϕ = $(join(round.(integrals; digits = 3), ", ")); ∫Σϕ = $(round(sum(integrals), digits = 3))"

f = Figure(size = (900, 800))
phase_observables = ntuple(k -> Observable(phases[k]), length(phases))
sum_observable = Observable(reduce(+, phases))
for k in eachindex(phases)
    row, column = divrem(k - 1, 2) .+ 1
    panel = f[row, column] = GridLayout()
    ax = Axis(panel[1, 1], title = "Phase $k", aspect = DataAspect())
    hm = heatmap!(ax, x, y, phase_observables[k]; colormap = cgrad(:roma, rev = true), colorrange = (0, 1))
    Colorbar(panel[1, 2], hm; label = "Phase $k")
end
sum_panel = f[2, 2] = GridLayout()
sum_axis = Axis(sum_panel[1, 1], title = sum_title(t, initial_integrals), aspect = DataAspect())
sum_heatmap = heatmap!(sum_axis, x, y, sum_observable; colormap = :balance, colorrange = (0.999999, 1.000001))
Colorbar(sum_panel[1, 2], sum_heatmap; label = "Sum of phases")
display(f)

while t < rotation_period
    Δt_step = min(Δt, rotation_period - t)
    WENO_step!(phases, velocity, scheme, Δt_step, Δx, Δy)
    global t += Δt_step
    global step += 1

    if step % 1000 == 0
        for k in eachindex(phases)
            phase_observables[k][] = phases[k]
        end
        current_integrals = map(sum, phases)
        sum_observable[] = reduce(+, phases)
        sum_axis.title = sum_title(t, current_integrals)
        sleep(0.01)
    end
end
for k in eachindex(phases)
    phase_observables[k][] = phases[k]
end
final_integrals = map(sum, phases)
sum_observable[] = reduce(+, phases)
sum_axis.title = sum_title(t, final_integrals)

simplex_error = maximum(abs, phases[1] .+ phases[2] .+ phases[3] .- 1)
bound_error = max(
    0.0,
    -minimum(minimum, phases),
    maximum(maximum, phases) - 1,
)

println("maximum simplex residual: ", simplex_error)
println("maximum bound violation:  ", bound_error)
for k in eachindex(phases)
    println("phase $k integral: $(initial_integrals[k]) -> $(final_integrals[k])")
end

## Convergence study ##########################################################
# Solid-body rotation is divergence-free (velocity is affine in x, y, so the
# two-point difference used for `scheme.divv` is exact), and one full period
# returns the composition to its initial state. The L1 error against that
# initial condition therefore measures the spatial truncation error of the
# WENO5-Z reconstruction directly, without the `ϕₖ∇·v` source term degrading it.

function rotation_initial_phases(nx, ny)
    Δx_, Δy_ = Lx / nx, Ly / ny
    x_ = (collect(1:nx) .- 0.5) .* Δx_
    y_ = (collect(1:ny) .- 0.5) .* Δy_
    w1_ = [0.05 + exp(-((xi - 0.32)^2 + (yj - 0.5)^2) / 0.015) for xi in x_, yj in y_]
    w2_ = [0.05 + exp(-((xi - 0.68)^2 + (yj - 0.5)^2) / 0.015) for xi in x_, yj in y_]
    w3_ = fill(0.15, nx, ny)
    total_ = w1_ .+ w2_ .+ w3_
    return (w1_ ./ total_, w2_ ./ total_, w3_ ./ total_), Δx_, Δy_
end

function rotation_error(nx, ny)
    ph, Δx_, Δy_ = rotation_initial_phases(nx, ny)
    exact = map(copy, ph)
    x_c = (collect(1:nx) .- 0.5) .* Δx_
    y_c = (collect(1:ny) .- 0.5) .* Δy_
    x_faces = range(0.0, Lx; length = nx + 1)
    y_faces = range(0.0, Ly; length = ny + 1)
    vel = (
        x = [-(yj - Ly / 2) for _ in x_faces, yj in y_c],
        y = [(xi - Lx / 2) for xi in x_c, _ in y_faces],
    )
    bnd = ntuple(_ -> PeriodicBC(), 4)
    sch = MultiphaseWENOScheme(ph; boundary = bnd, stag = true, multithreading = true)
    Δt_ = 0.25 * min(Δx_, Δy_)^(5 / 3)
    period = 2π
    t_ = 0.0
    while t_ < period
        step_ = min(Δt_, period - t_)
        WENO_step!(ph, vel, sch, step_, Δx_, Δy_)
        t_ += step_
    end
    return sum(k -> sum(abs, ph[k] .- exact[k]), eachindex(ph)) / (length(ph) * nx * ny)
end

resolutions = (32, 64, 128)
conv_errors = [rotation_error(n, n) for n in resolutions]
conv_rates = [log2(conv_errors[i] / conv_errors[i + 1]) for i in 1:(length(conv_errors) - 1)]
println("\nconvergence resolutions: ", resolutions)
println("convergence errors:      ", conv_errors)
println("convergence rates:       ", conv_rates)

conv_dx = collect(Lx ./ resolutions)
fig_conv = Figure(size = (500, 400))
ax_conv = Axis(
    fig_conv[1, 1]; xlabel = "Δx", ylabel = "L1 error", xscale = log10, yscale = log10,
    title = "Convergence, solid-body rotation (divergence-free)"
)
scatterlines!(ax_conv, conv_dx, conv_errors; label = "measured", marker = :circle)
ref_orders = (2, 3, 4, 5)
ref_styles = (:dash, :dashdot, :dashdotdot, :dot)
for (order, style) in zip(ref_orders, ref_styles)
    ref = conv_errors[1] .* (conv_dx ./ conv_dx[1]) .^ order
    lines!(ax_conv, conv_dx, ref; linestyle = style, label = "order $order")
end
axislegend(ax_conv; position = :rb)
display(fig_conv)
