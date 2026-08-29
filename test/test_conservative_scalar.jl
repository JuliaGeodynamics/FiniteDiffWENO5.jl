using Test
using FiniteDiffWENO5

# Manufactured smooth periodic data shared by every case below.
#   u(x) = 1 + 0.2 sin(2πx) + 0.1 cos(4πx)
#   v(x) = 1 + 0.3 sin(2πx)
# so the conservative flux is f = v u and
#   ∂ₓf = v' u + v u',  v' = 0.6π cos(2πx),
#   u' = 0.4π cos(2πx) − 0.4π sin(4πx).
manufactured_u(x) = 1 + 0.2 * sinpi(2x) + 0.1 * cospi(4x)
manufactured_v(x) = 1 + 0.3 * sinpi(2x)
manufactured_du(x) = 0.4π * cospi(2x) - 0.4π * sinpi(4x)
manufactured_dv(x) = 0.6π * cospi(2x)
manufactured_dflux(x) = manufactured_dv(x) * manufactured_u(x) + manufactured_v(x) * manufactured_du(x)

"""L1 error of the conservative operator against the analytic flux divergence."""
function conservative_error(n; stag)
    Δx = inv(float(n))
    x = ((1:n) .- 0.5) .* Δx
    u = manufactured_u.(x)
    boundary = (PeriodicBC(), PeriodicBC())
    weno = WENOScheme(u; form = :conservative, boundary, stag, multithreading = false)

    velocity = stag ?
        (; x = manufactured_v.((0:n) .* Δx)) :
        (; x = manufactured_v.(x))

    vcell = FiniteDiffWENO5.prepare_velocity!(weno, velocity)
    α = FiniteDiffWENO5.lf_speed(vcell.x)
    FiniteDiffWENO5.conservative_semi_discretisation_weno5!(
        weno.du, u, vcell, weno, n, inv(Δx), α,
    )
    return Δx * sum(abs, weno.du .- manufactured_dflux.(x)), weno
end

conservative_rates(; stag) =
    let errors = [first(conservative_error(n; stag)) for n in (64, 128, 256, 512)]
        log2.(errors[1:(end - 1)] ./ errors[2:end])
    end

@testset "conservative scalar transport" begin
    @testset "collocated velocity is fifth order" begin
        @test all(>(4.5), conservative_rates(stag = false))
    end

    @testset "staggered velocity is fifth order after ENO preparation" begin
        @test all(>(4.5), conservative_rates(stag = true))
    end

    @testset "periodic discrete integral is conserved" begin
        # A conservative operator must telescope: the summed divergence over a
        # periodic domain is zero independently of resolution. This is the
        # property the non-conservative material form does NOT have, and it is
        # the only reason to keep a conservative path at all.
        for n in (64, 128), stag in (false, true)
            _, weno = conservative_error(n; stag)
            @test abs(sum(weno.du)) < 128eps(Float64) * n
        end
    end

    @testset "constant velocity agrees with the material form" begin
        # For ∇·v = 0 the two PDE forms are identical, so the two discrete
        # operators must agree to roundoff. This catches sign and index errors
        # in the split-flux construction that a convergence test alone can miss.
        n = 128
        Δx = inv(float(n))
        x = ((1:n) .- 0.5) .* Δx
        u = manufactured_u.(x)
        boundary = (PeriodicBC(), PeriodicBC())
        velocity = (; x = fill(1.0, n))

        conservative = WENOScheme(u; form = :conservative, boundary, stag = false, multithreading = false)
        α = FiniteDiffWENO5.lf_speed(velocity.x)
        FiniteDiffWENO5.conservative_semi_discretisation_weno5!(
            conservative.du, u, velocity, conservative, n, inv(Δx), α,
        )

        material = WENOScheme(u; form = :nonconservative, boundary, stag = false, multithreading = false)
        FiniteDiffWENO5.WENO_flux!(material.fl, material.fr, u, material, n, 0.0, 0.0)
        FiniteDiffWENO5.material_semi_discretisation_weno5!(material.du, velocity, material, inv(Δx))

        # Both operators difference face values of the same magnitude and then
        # multiply by Δx⁻¹ = n, so the cancellation error they inherit grows
        # linearly with resolution. The measured gap is 2n·eps at every n, which
        # is roundoff, not disagreement; a sign or index error in the split flux
        # would instead produce an O(1) difference and still be caught here.
        @test conservative.du ≈ material.du rtol = 0 atol = 8 * n * eps(Float64)
    end
end

@testset "conservative scalar transport in 2D" begin
    # u  = 1 + 0.2 sin(2πx) cos(2πy)
    # vx = 1 + 0.3 sin(2πx),  vy = 0.5 + 0.2 cos(2πy)
    # ∂ₓ(vx u) + ∂y(vy u) = vx' u + vx u_x + vy' u + vy u_y
    u2(x, y) = 1 + 0.2 * sinpi(2x) * cospi(2y)
    vx2(x, y) = 1 + 0.3 * sinpi(2x)
    vy2(x, y) = 0.5 + 0.2 * cospi(2y)
    dflux2(x, y) =
        0.6π * cospi(2x) * u2(x, y) + vx2(x, y) * (0.4π * cospi(2x) * cospi(2y)) +
        (-0.4π * sinpi(2y)) * u2(x, y) + vy2(x, y) * (-0.4π * sinpi(2x) * sinpi(2y))

    function error2(n; stag)
        Δ = inv(float(n))
        x = ((1:n) .- 0.5) .* Δ
        faces = (0:n) .* Δ
        u = [u2(xi, yj) for xi in x, yj in x]
        boundary = ntuple(_ -> PeriodicBC(), 4)
        weno = WENOScheme(u; form = :conservative, boundary, stag, multithreading = false)
        velocity = stag ?
            (x = [vx2(xi, yj) for xi in faces, yj in x], y = [vy2(xi, yj) for xi in x, yj in faces]) :
            (x = [vx2(xi, yj) for xi in x, yj in x], y = [vy2(xi, yj) for xi in x, yj in x])
        vcell = FiniteDiffWENO5.prepare_velocity!(weno, velocity)
        αx, αy = FiniteDiffWENO5.lf_speed(vcell.x), FiniteDiffWENO5.lf_speed(vcell.y)
        FiniteDiffWENO5.conservative_semi_discretisation_weno5!(
            weno.du, u, vcell, weno, n, n, inv(Δ), inv(Δ), αx, αy,
        )
        return Δ^2 * sum(abs, weno.du .- [dflux2(xi, yj) for xi in x, yj in x]), weno
    end

    for stag in (false, true)
        errors = [first(error2(n; stag)) for n in (32, 64, 128, 256)]
        @test all(>(4.5), log2.(errors[1:(end - 1)] ./ errors[2:end]))
    end

    @testset "2D periodic integral is conserved" begin
        for stag in (false, true)
            _, weno = error2(64; stag)
            @test abs(sum(weno.du)) < 1.0e-9
        end
    end
end
