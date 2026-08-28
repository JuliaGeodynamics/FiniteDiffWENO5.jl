using FiniteDiffWENO5
using GLMakie

nx, ny = 96, 96
Lx, Ly = 1.0, 1.0
Δx, Δy = Lx / nx, Ly / ny
x = (collect(1:nx) .- 0.5) .* Δx
y = (collect(1:ny) .- 0.5) .* Δy

# A smooth three-material composition. Normalising positive marker functions gives
# pointwise fractions in [0, 1] whose sum is one.
w1 = [0.05 + exp(-((xi - 0.32)^2 + (yj - 0.50)^2) / 0.015) for xi in x, yj in y]
w2 = [0.05 + exp(-((xi - 0.68)^2 + (yj - 0.50)^2) / 0.015) for xi in x, yj in y]
w3 = fill(0.15, nx, ny)
total = w1 .+ w2 .+ w3
phases = (w1 ./ total, w2 ./ total, w3 ./ total)

# Start from the staggered deformation flow in 2D_linear_deformation.jl and add
# a vertical compression term. Its divergence is 2π * compression * cos(2πy).
x_vx = range(0.0, Lx; length = nx + 1)
y_vx = range(Δy / 2, Ly - Δy / 2; length = ny)
x_vy = range(Δx / 2, Lx - Δx / 2; length = nx)
y_vy = range(0.0, Ly; length = ny + 1)
X_vx = repeat(x_vx, 1, ny)
Y_vx = repeat(y_vx', nx + 1, 1)
X_vy = repeat(x_vy, 1, ny + 1)
Y_vy = repeat(y_vy', nx, 1)

compression = 0.5
velocity = (
    x = -2π .* sin.(π .* X_vx) .* cos.(π .* Y_vx),
    y = 2π .* cos.(π .* X_vy) .* sin.(π .* Y_vy) .+
        compression .* sin.(2π .* Y_vy),
)
boundary = ntuple(_ -> ExtrapolateBC(), 4)
scheme = MultiphaseWENOScheme(
    phases; boundary = boundary, stag = true, multithreading = true)

initial_integrals = map(sum, phases)
Δt = 0.25 * min(Δx, Δy)^(5 / 3)
tmax = 1.0
t = 0.0
step = 0
reversed = false

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

while t < tmax
    Δt_step = min(Δt, tmax - t)
    WENO_step!(phases, velocity, scheme, Δt_step, Δx, Δy)
    global t += Δt_step
    global step += 1

    if !reversed && t >= tmax / 2
        velocity.x .*= -1
        velocity.y .*= -1
        global reversed = true
    end

    if step % 100 == 0
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
# Reversing the deformation flow at t = tmax/2 returns the composition to its
# initial state, so the L1 error at t = tmax again measures spatial truncation
# error directly. Unlike the rotation case, this flow has nonzero local
# divergence (the compression term), and `scheme.divv` only approximates it to
# second order via a two-point difference. That O(Δx²) source-term error
# dominates the O(Δx⁵) WENO flux error as Δx → 0, so convergence here is
# expected to be second order rather than fifth order.

function deformation_initial_phases(nx, ny)
    Δx_, Δy_ = Lx / nx, Ly / ny
    x_ = (collect(1:nx) .- 0.5) .* Δx_
    y_ = (collect(1:ny) .- 0.5) .* Δy_
    w1_ = [0.05 + exp(-((xi - 0.32)^2 + (yj - 0.50)^2) / 0.015) for xi in x_, yj in y_]
    w2_ = [0.05 + exp(-((xi - 0.68)^2 + (yj - 0.50)^2) / 0.015) for xi in x_, yj in y_]
    w3_ = fill(0.15, nx, ny)
    total_ = w1_ .+ w2_ .+ w3_
    return (w1_ ./ total_, w2_ ./ total_, w3_ ./ total_), Δx_, Δy_
end

# function deformation_error(nx, ny)
#     ph, Δx_, Δy_ = deformation_initial_phases(nx, ny)
#     exact = map(copy, ph)
#     x_vx_ = range(0.0, Lx; length = nx + 1)
#     y_vx_ = range(Δy_ / 2, Ly - Δy_ / 2; length = ny)
#     x_vy_ = range(Δx_ / 2, Lx - Δx_ / 2; length = nx)
#     y_vy_ = range(0.0, Ly; length = ny + 1)
#     X_vx_ = repeat(x_vx_, 1, ny)
#     Y_vx_ = repeat(y_vx_', nx + 1, 1)
#     X_vy_ = repeat(x_vy_, 1, ny + 1)
#     Y_vy_ = repeat(y_vy_', nx, 1)
#     vel = (
#         x = -2π .* sin.(π .* X_vx_) .* cos.(π .* Y_vx_),
#         y = 2π .* cos.(π .* X_vy_) .* sin.(π .* Y_vy_) .+
#             compression .* sin.(2π .* Y_vy_),
#     )
#     bnd = ntuple(_ -> ExtrapolateBC(), 4)
#     sch = MultiphaseWENOScheme(ph; boundary = bnd, stag = true, multithreading = true)
#     Δt_ = 0.25 * min(Δx_, Δy_)^(5 / 3)
#     t_ = 0.0
#     rev = false
#     while t_ < tmax
#         step_ = min(Δt_, tmax - t_)
#         WENO_step!(ph, vel, sch, step_, Δx_, Δy_)
#         t_ += step_
#         if !rev && t_ >= tmax / 2
#             vel.x .*= -1
#             vel.y .*= -1
#             rev = true
#         end
#     end
#     return sum(k -> sum(abs, ph[k] .- exact[k]), eachindex(ph)) / (length(ph) * nx * ny)
# end

# resolutions = (32, 64, 128)
# conv_errors = [deformation_error(n, n) for n in resolutions]
# conv_rates = [log2(conv_errors[i] / conv_errors[i + 1]) for i in 1:(length(conv_errors) - 1)]
# println("\nconvergence resolutions: ", resolutions)
# println("convergence errors:      ", conv_errors)
# println("convergence rates:       ", conv_rates)

# conv_dx = collect(Lx ./ resolutions)
# fig_conv = Figure(size = (500, 400))
# ax_conv = Axis(
#     fig_conv[1, 1]; xlabel = "Δx", ylabel = "L1 error", xscale = log10, yscale = log10,
#     title = "Convergence, reversed deformation flow (∇·v ≠ 0)")
# scatterlines!(ax_conv, conv_dx, conv_errors; label = "measured", marker = :circle)
# ref_orders = (2, 3, 4, 5)
# ref_styles = (:dash, :dashdot, :dashdotdot, :dot)
# for (order, style) in zip(ref_orders, ref_styles)
#     ref = conv_errors[1] .* (conv_dx ./ conv_dx[1]) .^ order
#     lines!(ax_conv, conv_dx, ref; linestyle = style, label = "order $order")
# end
# axislegend(ax_conv; position = :rb)
# display(fig_conv)
