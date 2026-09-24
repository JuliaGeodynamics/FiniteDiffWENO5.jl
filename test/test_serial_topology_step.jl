using Test
using FiniteDiffWENO5
using FiniteDiffWENO5: NoTopology, SerialTopology
using KernelAbstractions

step_detfield(n) = [1.0 + 0.3sinpi(2 * (i - 0.5) / n) + 0.15cospi(4 * (i - 0.5) / n) for i in 1:n]
step_detfield(n, m) = [1.0 + 0.3sinpi(2 * (i - 0.5) / n) * cospi(2 * (j - 0.5) / m) for i in 1:n, j in 1:m]
step_detfield(n, m, p) = [1.0 + 0.2sinpi(2 * (i - 0.5) / n) * cospi(2 * (j - 0.5) / m) * sinpi(2 * (k - 0.5) / p) for i in 1:n, j in 1:m, k in 1:p]
step_detface(n) = [1.0 + 0.4sinpi(2 * i / n) for i in 0:n]

function step_boundary(kind)
    kind === :extrapolate && return (ExtrapolateBC(), ExtrapolateBC())
    kind === :periodic && return (PeriodicBC(), PeriodicBC())
    kind === :inflow && return (PrescribedInflowBC(0.73), PrescribedInflowBC(1.61))
    error("unknown boundary kind $kind")
end

@testset "SerialTopology reproduces serial bit-for-bit" begin

    @testset "1D: $bk, stag=$stag, $form" for bk in (:extrapolate, :periodic, :inflow),
        stag in (false, true), form in (:nonconservative, :conservative)

        n = 20
        halo = 3
        dt, dx = 0.001, inv(n)
        boundary = step_boundary(bk)
        u = step_detfield(n)

        weno_s = WENOScheme(copy(u); boundary, form, stag, multithreading = false)
        vel_s = stag ? (; x = step_detface(n)) : (; x = step_detfield(n) .+ 0.5)
        u_s = copy(u)
        WENO_step!(u_s, vel_s, weno_s, dt, dx)

        topo = SerialTopology((n,); halo, periodic = bk === :periodic)
        c0 = zeros(n + 2halo)
        c0[(halo + 1):(halo + n)] .= u
        weno_t = WENOScheme(c0, topo; boundary, form, stag, multithreading = false)

        vel_t = if stag
            vf = zeros(n + 2halo + 1)
            vf[(halo + 1):(halo + n + 1)] .= vel_s.x
            (; x = vf)
        else
            vc = zeros(n + 2halo)
            vc[(halo + 1):(halo + n)] .= vel_s.x
            (; x = vc)
        end
        WENO_step!(c0, vel_t, weno_t, dt, dx)

        @test c0[(halo + 1):(halo + n)] == u_s
    end

    @testset "2D: $bk, stag=$stag, $form" for bk in (:extrapolate, :periodic),
        stag in (false, true), form in (:nonconservative, :conservative)

        nx, ny = 12, 10
        halo = 3
        dt, dx, dy = 0.001, inv(nx), inv(ny)
        boundary = ntuple(_ -> step_boundary(bk)[1], 4)
        u = step_detfield(nx, ny)

        weno_s = WENOScheme(copy(u); boundary, form, stag, multithreading = false)
        vel_s = if stag
            (x = [1.0 + 0.3sinpi(2i / nx) for i in 0:nx, j in 1:ny], y = [0.5 + 0.2cospi(2j / ny) for i in 1:nx, j in 0:ny])
        else
            (x = step_detfield(nx, ny) .+ 0.4, y = step_detfield(nx, ny) .+ 0.1)
        end
        u_s = copy(u)
        WENO_step!(u_s, vel_s, weno_s, dt, dx, dy)

        topo = SerialTopology((nx, ny); halo, periodic = bk === :periodic)
        c0 = zeros(nx + 2halo, ny + 2halo)
        c0[(halo + 1):(halo + nx), (halo + 1):(halo + ny)] .= u
        weno_t = WENOScheme(c0, topo; boundary, form, stag, multithreading = false)

        vel_t = if stag
            vx = zeros(nx + 2halo + 1, ny + 2halo)
            vy = zeros(nx + 2halo, ny + 2halo + 1)
            vx[(halo + 1):(halo + nx + 1), (halo + 1):(halo + ny)] .= vel_s.x
            vy[(halo + 1):(halo + nx), (halo + 1):(halo + ny + 1)] .= vel_s.y
            (; x = vx, y = vy)
        else
            vx = zeros(nx + 2halo, ny + 2halo)
            vy = zeros(nx + 2halo, ny + 2halo)
            vx[(halo + 1):(halo + nx), (halo + 1):(halo + ny)] .= vel_s.x
            vy[(halo + 1):(halo + nx), (halo + 1):(halo + ny)] .= vel_s.y
            (; x = vx, y = vy)
        end
        WENO_step!(c0, vel_t, weno_t, dt, dx, dy)

        @test c0[(halo + 1):(halo + nx), (halo + 1):(halo + ny)] == u_s
    end

    @testset "3D smoke test: ExtrapolateBC, stag=true, nonconservative" begin
        n = 8
        halo = 3
        dt, dx = 0.001, inv(n)
        boundary = ntuple(_ -> ExtrapolateBC(), 6)
        u = step_detfield(n, n, n)

        weno_s = WENOScheme(copy(u); boundary, form = :nonconservative, stag = true, multithreading = false)
        vel_s = (
            x = [1.0 + 0.3sinpi(2i / n) for i in 0:n, j in 1:n, k in 1:n],
            y = [0.5 + 0.2cospi(2j / n) for i in 1:n, j in 0:n, k in 1:n],
            z = [0.4 + 0.1sinpi(2k / n) for i in 1:n, j in 1:n, k in 0:n],
        )
        u_s = copy(u)
        WENO_step!(u_s, vel_s, weno_s, dt, dx, dx, dx)

        topo = SerialTopology((n, n, n); halo)
        c0 = zeros(n + 2halo, n + 2halo, n + 2halo)
        r = (halo + 1):(halo + n)
        c0[r, r, r] .= u
        weno_t = WENOScheme(c0, topo; boundary, form = :nonconservative, stag = true, multithreading = false)

        vx = zeros(n + 2halo + 1, n + 2halo, n + 2halo)
        vy = zeros(n + 2halo, n + 2halo + 1, n + 2halo)
        vz = zeros(n + 2halo, n + 2halo, n + 2halo + 1)
        vx[(halo + 1):(halo + n + 1), r, r] .= vel_s.x
        vy[r, (halo + 1):(halo + n + 1), r] .= vel_s.y
        vz[r, r, (halo + 1):(halo + n + 1)] .= vel_s.z
        WENO_step!(c0, (; x = vx, y = vy, z = vz), weno_t, dt, dx, dx, dx)

        @test c0[r, r, r] == u_s
    end

    @testset "1D tuple path: extrapolate, stag=true, conservative" begin
        n = 16
        halo = 3
        dt, dx = 0.001, inv(n)
        boundary = (ExtrapolateBC(), ExtrapolateBC())
        u1, u2 = step_detfield(n), step_detfield(n) .+ 0.3
        vface = step_detface(n)

        weno_s = WENOScheme(copy(u1); boundary, form = :conservative, stag = true, multithreading = false)
        u1s, u2s = copy(u1), copy(u2)
        WENO_step!((u1s, u2s), (; x = vface), weno_s, dt, dx; u_min = (0.0, 0.0), u_max = (2.0, 2.0))

        topo = SerialTopology((n,); halo)
        c1 = zeros(n + 2halo)
        c2 = zeros(n + 2halo)
        c1[(halo + 1):(halo + n)] .= u1
        c2[(halo + 1):(halo + n)] .= u2
        weno_t = WENOScheme(c1, topo; boundary, form = :conservative, stag = true, multithreading = false)
        vf = zeros(n + 2halo + 1)
        vf[(halo + 1):(halo + n + 1)] .= vface
        WENO_step!((c1, c2), (; x = vf), weno_t, dt, dx; u_min = (0.0, 0.0), u_max = (2.0, 2.0))

        @test c1[(halo + 1):(halo + n)] == u1s
        @test c2[(halo + 1):(halo + n)] == u2s
    end

    @testset "upwind_mode = true with a topology is rejected at construction" begin
        n = 10
        halo = 3
        topo = SerialTopology((n,); halo)
        c0 = zeros(n + 2halo)
        @test_throws ArgumentError WENOScheme(c0, topo; boundary = (ExtrapolateBC(), ExtrapolateBC()), form = :nonconservative, upwind_mode = true)
    end

    @testset "KA WENO_step! on a scheme carrying a topology throws a clear error" begin
        n = 12
        halo = 3
        topo = SerialTopology((n,); halo)
        c0 = zeros(n + 2halo)
        weno = WENOScheme(c0, topo; boundary = (ExtrapolateBC(), ExtrapolateBC()), form = :nonconservative, stag = false, multithreading = false)
        u = zeros(n + 2halo)
        v = (; x = zeros(n + 2halo))
        @test_throws ArgumentError WENO_step!(u, v, weno, 0.001, inv(n), CPU())
    end

    @testset "construction size mismatch throws DimensionMismatch naming both shapes" begin
        n = 10
        halo = 3
        topo = SerialTopology((n,); halo)
        bad = zeros(n) # missing the padding entirely
        @test_throws DimensionMismatch WENOScheme(bad, topo; boundary = (ExtrapolateBC(), ExtrapolateBC()), form = :nonconservative)
    end

    # WENO5 requires a halo of at least three cells for reconstruction.
    @testset "halo < 3 is rejected at scheme construction, not left to silently misreconstruct" for halo in (1, 2)
        n = 20
        topo = SerialTopology((n,); halo)
        c0 = zeros(n + 2halo)
        @test_throws ArgumentError WENOScheme(
            c0, topo; boundary = (ExtrapolateBC(), ExtrapolateBC()), form = :nonconservative
        )

        p = ntuple(_ -> zeros(n + 2halo), 3)
        @test_throws ArgumentError MultiphaseWENOScheme(
            p, topo; boundary = (ExtrapolateBC(), ExtrapolateBC())
        )
    end

    @testset "halo == 3 is accepted (boundary case of the check above)" begin
        n = 20
        halo = 3
        topo = SerialTopology((n,); halo)
        c0 = zeros(n + 2halo)
        weno = WENOScheme(c0, topo; boundary = (ExtrapolateBC(), ExtrapolateBC()), form = :nonconservative)
        @test weno isa WENOScheme
    end
end
