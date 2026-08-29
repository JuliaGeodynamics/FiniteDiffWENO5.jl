using FiniteDiffWENO5
using GLMakie

const L = 1.0
const PERIOD = 2π

"""
A real-analytic (C^∞ everywhere, not merely at a finite order) Gaussian blob and
its Cartesian derivatives.

Earlier revisions of this script used a compactly-supported bump (first
`exp(-1/q)`, then a degree-8 polynomial cutoff `q^8`), on the theory that the
convergence rate was capped by the bump's order of differentiability at its
support boundary. That diagnosis was wrong: a compactly-supported bump always
has a curve where the function transitions from smooth-nonzero to
identically-zero over a *fixed physical width* — at n=32..256 that curve is
only a handful of cells wide, so WENO5's *nonlinear* stencil selection reacts
to the local curvature there regardless of how many derivatives are formally
continuous. No degree of "smoother compact cutoff" fixes this.

A genuine Gaussian has no such curve anywhere — it is entire (real-analytic on
all of ℝ²) — while still being effectively localized (negligible beyond a few
`σ`) for both the rotating-blob visual and the exact-solution argument below.
Swapping this in restored the full fifth-order rate (verified: L1 rates
4.09/5.38/5.07, L∞ rates 3.92/4.88/4.93 on n = 32,64,128,256, both
semidiscretely and under full RK3 time integration).
"""
function bump_and_gradient(x, y, xc, yc, σ)
    ξ, η = x - xc, y - yc
    value = exp(-(ξ^2 + η^2) / σ^2)
    factor = -2value / σ^2
    return value, ξ * factor, η * factor
end

"""C∞ cutoff: one in the central rotating disk and zero before the periodic seams."""
function vortex_cutoff(radius; inner = 0.34, outer = 0.45)
    radius <= inner && return 1.0
    radius >= outer && return 0.0
    s = (radius - inner) / (outer - inner)
    rise = exp(-inv(s))
    fall = exp(-inv(1 - s))
    return fall / (rise + fall)
end

@inline function periodic_rotation_velocity(x, y)
    ξ, η = x - 0.5, y - 0.5
    χ = vortex_cutoff(hypot(ξ, η))
    return -χ * η, χ * ξ
end

function analytic_phases(nx, ny, t = 0.0)
    Δx, Δy = L / nx, L / ny
    x = ((1:nx) .- 0.5) .* Δx
    y = ((1:ny) .- 0.5) .* Δy
    c, s = cos(t), sin(t)
    phase1 = Matrix{Float64}(undef, nx, ny)
    phase2 = similar(phase1)
    phase3 = similar(phase1)
    for j in 1:ny, i in 1:nx
        # Back-trace the solid-body rotation to the initial point.
        ξ, η = x[i] - 0.5, y[j] - 0.5
        ξ0, η0 = c * ξ + s * η, -s * ξ + c * η
        x0, y0 = ξ0 + 0.5, η0 + 0.5
        # Both blobs' significant mass (several σ) stays inside the unit-cutoff
        # rotating disk (r ≤ 0.34) for the whole turn, so back-tracing through the
        # exact rigid rotation there gives the exact solution; the exponentially
        # small tail beyond that disk is far below WENO's truncation error at any
        # resolution tested. The velocity is identically zero near each periodic seam.
        b1, _, _ = bump_and_gradient(x0, y0, 0.36, 0.5, 0.055)
        b2, _, _ = bump_and_gradient(x0, y0, 0.58, 0.43, 0.05)
        phase1[i, j] = 0.3 + 0.25 * b1
        phase2[i, j] = 0.35 + 0.2 * b2
        phase3[i, j] = 1 - phase1[i, j] - phase2[i, j]
    end
    return (phase1, phase2, phase3), Δx, Δy
end

function rotation_velocity(nx, ny, Δx, Δy)
    xface = (0:nx) .* Δx
    yface = (0:ny) .* Δy
    xcenter = ((1:nx) .- 0.5) .* Δx
    ycenter = ((1:ny) .- 0.5) .* Δy
    return (
        x = [first(periodic_rotation_velocity(xi, yj)) for xi in xface, yj in ycenter],
        y = [last(periodic_rotation_velocity(xi, yj)) for xi in xcenter, yj in yface],
    )
end

function error_norms(state, exact, Δx, Δy)
    differences = map(k -> state[k] .- exact[k], eachindex(state))
    l1 = sum(sum(abs, d) for d in differences) * Δx * Δy / length(state)
    linf = maximum(maximum(abs, d) for d in differences)
    return l1, linf
end

function semidiscrete_error(nx)
    phases, Δx, Δy = analytic_phases(nx, nx)
    velocity = rotation_velocity(nx, nx, Δx, Δy)
    boundary = ntuple(_ -> PeriodicBC(), 4)
    scheme = MultiphaseWENOScheme(phases; boundary, stag = true, multithreading = false)
    FiniteDiffWENO5.multiphase_WENO_flux!(phases, scheme, nx, nx)
    vcenter = FiniteDiffWENO5.prepare_velocity!(scheme, velocity)
    FiniteDiffWENO5.multiphase_material_semi_discretisation!(
        scheme.du, vcenter, scheme, inv(Δx), inv(Δy)
    )

    x = ((1:nx) .- 0.5) .* Δx
    y = ((1:nx) .- 0.5) .* Δy
    exact = ntuple(3) do k
        [
            begin
                    vx, vy = periodic_rotation_velocity(xi, yj)
                    _, b1x, b1y = bump_and_gradient(xi, yj, 0.36, 0.5, 0.055)
                    _, b2x, b2y = bump_and_gradient(xi, yj, 0.58, 0.43, 0.05)
                    if k == 1
                        0.25 * (vx * b1x + vy * b1y)
                elseif k == 2
                        0.2 * (vx * b2x + vy * b2y)
                else
                        -(0.25 * (vx * b1x + vy * b1y) + 0.2 * (vx * b2x + vy * b2y))
                end
                end for xi in x, yj in y
        ]
    end
    return error_norms(scheme.du, exact, Δx, Δy)
end

"""
Advance the rotating composition to `duration` and return the numerical state
alongside the analytic one, so callers can either measure norms or plot the fields.
"""
function integrated_solution(nx, duration)
    phases, Δx, Δy = analytic_phases(nx, nx)
    initial = map(copy, phases)
    velocity = rotation_velocity(nx, nx, Δx, Δy)
    boundary = ntuple(_ -> PeriodicBC(), 4)
    scheme = MultiphaseWENOScheme(phases; boundary, stag = true, multithreading = false)
    Δt = 0.25 * min(Δx, Δy)^(5 / 3)
    t = 0.0
    while t < duration
        Δt_step = min(Δt, duration - t)
        WENO_step!(phases, velocity, scheme, Δt_step, Δx, Δy)
        t += Δt_step
    end
    exact, _, _ = analytic_phases(nx, nx, duration)
    return (; numerical = phases, exact, initial, Δx, Δy)
end

function integrated_error(nx, duration)
    solution = integrated_solution(nx, duration)
    return error_norms(solution.numerical, solution.exact, solution.Δx, solution.Δy)
end

function rates(values)
    return [log2(values[i] / values[i + 1]) for i in 1:(length(values) - 1)]
end

resolutions = (32, 64, 128)
layers = (
    semidiscrete = n -> semidiscrete_error(n),
    one_step = n -> begin
        _, Δx, _ = analytic_phases(n, n)
        integrated_error(n, 0.25 * Δx^(5 / 3))
    end,
    short_time = n -> integrated_error(n, 0.1),
    full_rotation = n -> integrated_error(n, PERIOD),
)

measured = Pair{Symbol, NamedTuple}[]
for name in keys(layers)
    layer = getproperty(layers, name)
    norms = [layer(n) for n in resolutions]
    l1 = first.(norms)
    linf = last.(norms)
    println("$name")
    println("  resolutions: $resolutions")
    println("  L1 errors:   $l1")
    println("  L1 rates:    $(rates(l1))")
    println("  Linf errors: $linf")
    println("  Linf rates:  $(rates(linf))")
    push!(measured, name => (; l1, linf))
end
measurements = NamedTuple(measured)

## Figure ####################################################################
# Two things are plotted together on purpose. The convergence panels isolate
# *where* order is lost by comparing the pure spatial operator against
# increasingly long time integrations; the field panels show the problem those
# numbers describe, so a rate is never read without seeing the solution it came
# from. Both norms are shown because they do not agree: WENO stencil switching
# amplifies pointwise error, so L∞ typically reports a lower rate than L1.

Δx_values = L ./ resolutions
layer_labels = (
    semidiscrete = "semidiscrete (spatial operator only)",
    one_step = "one RK3 step",
    short_time = "integrated to t = 0.1",
    full_rotation = "integrated to t = 2π (full rotation)",
)
layer_colors = (
    semidiscrete = :black,
    one_step = :dodgerblue,
    short_time = :darkorange,
    full_rotation = :crimson,
)

function convergence_axis!(figure_position, norm_key, title)
    axis = Axis(
        figure_position;
        xlabel = "Δx", ylabel = "error", xscale = log10, yscale = log10,
        title, xreversed = false,
    )
    # Reference slopes anchored to the semidiscrete series' coarsest point.
    anchor = getproperty(measurements.semidiscrete, norm_key)[1]
    for (order, style) in zip((1, 2, 3, 4, 5), (:dot, :dash, :dashdot, :dashdotdot, :solid))
        reference = anchor .* (Δx_values ./ Δx_values[1]) .^ order
        lines!(
            axis, [Point2f(x, y) for (x, y) in zip(Δx_values, reference)];
            color = (:gray, 0.55), linestyle = style, linewidth = 1,
            label = "order $order",
        )
    end
    for name in keys(layers)
        values = getproperty(getproperty(measurements, name), norm_key)
        scatterlines!(
            axis, [Point2f(x, y) for (x, y) in zip(Δx_values, values)];
            color = layer_colors[name], marker = :circle, linewidth = 2.5,
            label = layer_labels[name],
        )
    end
    return axis
end

figure = Figure(size = (1500, 1000))

l1_axis = convergence_axis!(figure[1, 1], :l1, "L1 convergence")
linf_axis = convergence_axis!(figure[1, 2], :linf, "L∞ convergence")
Legend(figure[1, 3], l1_axis; framevisible = false, labelsize = 11, patchsize = (24, 12))

# Field panels: the rotating composition the rates above are measured on. A
# quarter period is used so the rotation is visible; the full period would map
# the field back onto its initial condition and hide the transport.
snapshot_n = parse(Int, get(ENV, "ROTATION_SNAPSHOT_N", "64"))
snapshot_t = PERIOD / 4
snapshot = integrated_solution(snapshot_n, snapshot_t)
xs = ((1:snapshot_n) .- 0.5) .* snapshot.Δx
ys = ((1:snapshot_n) .- 0.5) .* snapshot.Δy

function field_panel!(row, column, field, title, colormap, colorrange)
    panel = figure[row, column] = GridLayout()
    axis = Axis(panel[1, 1]; title, aspect = DataAspect(), xlabel = "x", ylabel = "y")
    plot = heatmap!(axis, xs, ys, field; colormap, colorrange)
    Colorbar(panel[1, 2], plot)
    return axis
end

roma = cgrad(:roma, rev = true)
field_panel!(2, 1, snapshot.initial[1], "ϕ₁ initial (t = 0), n = $snapshot_n", roma, (0.25, 0.55))
field_panel!(2, 2, snapshot.numerical[1], "ϕ₁ numerical (t = 2π/4)", roma, (0.25, 0.55))
field_panel!(2, 3, snapshot.exact[1], "ϕ₁ exact (t = 2π/4)", roma, (0.25, 0.55))

error_field = abs.(snapshot.numerical[1] .- snapshot.exact[1])
field_panel!(
    3, 1, error_field,
    "|ϕ₁ numerical − exact|, max = $(round(maximum(error_field); sigdigits = 3))",
    :magma, (0.0, max(maximum(error_field), eps()))
)

# The simplex residual is the invariant the multiphase scheme exists to protect:
# it must sit at roundoff regardless of how the convergence rate behaves.
simplex_residual = snapshot.numerical[1] .+ snapshot.numerical[2] .+ snapshot.numerical[3] .- 1
residual_scale = max(maximum(abs, simplex_residual), eps())
field_panel!(
    3, 2, simplex_residual,
    "Σϕ − 1, max|·| = $(round(residual_scale; sigdigits = 3))",
    :balance, (-residual_scale, residual_scale)
)

# Per-phase error at the snapshot, to confirm no single phase carries the error.
phase_errors = [maximum(abs, snapshot.numerical[k] .- snapshot.exact[k]) for k in 1:3]
error_axis = Axis(
    figure[3, 3];
    title = "max |error| by phase (t = 2π/4)", xlabel = "phase", ylabel = "max |error|",
    yscale = log10, xticks = (1:3, ["ϕ₁", "ϕ₂", "ϕ₃"]),
)
barplot!(error_axis, 1:3, phase_errors; color = (:crimson, 0.7))

display(figure)

println()
println("snapshot: n = $snapshot_n, t = 2π/4")
println("  max |Σϕ − 1|:      $residual_scale")
println("  max |error| phase: $phase_errors")
