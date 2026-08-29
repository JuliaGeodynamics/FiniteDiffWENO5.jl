using FiniteDiffWENO5

"""Semidiscrete fifth-order check for multiphase material advection.

This isolates the spatial operator using smooth periodic phase fractions and a
smooth, genuinely compressible velocity supplied on staggered faces.
"""
function run_case(n)
    dx = 1 / n
    x = (collect(1:n) .- 0.5) .* dx
    X = repeat(x, 1, n)
    Y = repeat(x', n, 1)

    p1 = @. 0.3 + 0.07 * sinpi(2X) * cospi(2Y)
    p2 = @. 0.35 + 0.06 * cospi(2X) * sinpi(2Y)
    phases = (p1, p2, 1 .- p1 .- p2)

    xf = range(0, 1, length = n + 1)
    Xfx = repeat(xf, 1, n)
    Yfy = repeat(xf', n, 1)
    velocity = (
        x = 0.7 .+ 0.15 .* sinpi.(2 .* Xfx),
        y = -0.4 .+ 0.12 .* cospi.(2 .* Yfy),
    )
    vx = @. 0.7 + 0.15 * sinpi(2X)
    vy = @. -0.4 + 0.12 * cospi(2Y)

    boundary = ntuple(_ -> PeriodicBC(), 4)
    scheme = MultiphaseWENOScheme(phases; boundary, stag = true, multithreading = false)
    FiniteDiffWENO5.multiphase_WENO_flux!(phases, scheme, n, n)
    if isdefined(FiniteDiffWENO5, :prepare_velocity!)
        vcenter = FiniteDiffWENO5.prepare_velocity!(scheme, velocity)
        FiniteDiffWENO5.multiphase_material_semi_discretisation!(
            scheme.du, vcenter, scheme, inv(dx), inv(dx)
        )
    else
        # Compatibility branch for the pre-ENO implementation. It uses face fluxes
        # and a two-point divergence/source cancellation, which is the baseline this
        # example is intended to quantify.
        FiniteDiffWENO5.multiphase_semi_discretisation!(
            scheme.du, phases, velocity, scheme, inv(dx), inv(dx)
        )
    end

    dp1dx = @. 0.14π * cospi(2X) * cospi(2Y)
    dp1dy = @. -0.14π * sinpi(2X) * sinpi(2Y)
    dp2dx = @. -0.12π * sinpi(2X) * sinpi(2Y)
    dp2dy = @. 0.12π * cospi(2X) * cospi(2Y)
    exact = (
        vx .* dp1dx .+ vy .* dp1dy,
        vx .* dp2dx .+ vy .* dp2dy,
        .-vx .* (dp1dx .+ dp2dx) .- vy .* (dp1dy .+ dp2dy),
    )
    return sum(k -> sum(abs, scheme.du[k] .- exact[k]), 1:3) / (3n^2)
end

resolutions = (24, 48, 96)
errors = [run_case(n) for n in resolutions]
rates = [log2(errors[i] / errors[i + 1]) for i in 1:(length(errors) - 1)]
println("compressible multiphase semidiscrete convergence")
println("resolutions: ", resolutions)
println("errors:      ", errors)
println("rates:       ", rates)
