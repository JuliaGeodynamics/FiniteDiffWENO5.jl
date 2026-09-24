using Test
using MPI
using FiniteDiffWENO5
using FiniteDiffWENO5: weno_cartesian_topology, weno_owned_size, weno_global_offset,
    owned_window, lf_speed

const comm = MPI.COMM_WORLD
const nprocs = MPI.Comm_size(comm)
const rank = MPI.Comm_rank(comm)

# A smooth, deterministic function of GLOBAL coordinates: every rank (and
# the redundantly-computed serial reference) evaluates the exact same
# formula, so there is no communication needed to agree on the input data.
gfield(gi, n) = 1.0 + 0.3sinpi(2 * (gi - 0.5) / n) + 0.15cospi(4 * (gi - 0.5) / n)
gfield(gi, gj, n, m) = 1.0 + 0.3sinpi(2 * (gi - 0.5) / n) * cospi(2 * (gj - 0.5) / m)
gface(gi, n) = 1.0 + 0.4sinpi(2 * gi / n)

function serial_reference_1d(n, form, stag, boundary; dt, dx)
    u = [gfield(i, n) for i in 1:n]
    weno = WENOScheme(copy(u); boundary, form, stag, multithreading = false)
    vel = stag ? (; x = [gface(i, n) for i in 0:n]) : (; x = [gfield(i, n) + 0.5 for i in 1:n])
    WENO_step!(u, vel, weno, dt, dx)
    return u
end

function serial_reference_2d(nx, ny, form, boundary; dt, dx, dy)
    u = [gfield(i, j, nx, ny) for i in 1:nx, j in 1:ny]
    weno = WENOScheme(copy(u); boundary, form, stag = false, multithreading = false)
    vel = (x = [gfield(i, j, nx, ny) + 0.4 for i in 1:nx, j in 1:ny], y = [gfield(i, j, nx, ny) + 0.1 for i in 1:nx, j in 1:ny])
    WENO_step!(u, vel, weno, dt, dx, dy)
    return u
end

@testset "distributed scalar WENO_step!" begin

    @testset "1D bit-for-bit: $bk, $form, stag=false" for bk in (:extrapolate, :periodic),
            form in (:nonconservative, :conservative)

        n = 40
        halo = 3
        dt, dx = 0.001, inv(n)
        boundary = bk === :extrapolate ? (ExtrapolateBC(), ExtrapolateBC()) : (PeriodicBC(), PeriodicBC())

        u_s = serial_reference_1d(n, form, false, boundary; dt, dx)

        topo = weno_cartesian_topology((n,); comm, halo, periodic = bk === :periodic)
        owned = weno_owned_size(topo)
        offset = weno_global_offset(topo)

        c0 = zeros(owned[1] + 2halo)
        c0[(halo + 1):(halo + owned[1])] .= [gfield(offset[1] + i, n) for i in 1:owned[1]]
        weno = WENOScheme(c0, topo; boundary, form, stag = false, multithreading = false)
        v = zeros(owned[1] + 2halo)
        v[(halo + 1):(halo + owned[1])] .= [gfield(offset[1] + i, n) + 0.5 for i in 1:owned[1]]

        WENO_step!(c0, (; x = v), weno, dt, dx)

        got = c0[(halo + 1):(halo + owned[1])]
        expected = u_s[(offset[1] + 1):(offset[1] + owned[1])]
        @test got == expected
    end

    @testset "2D bit-for-bit: $bk, $form, stag=false" for bk in (:extrapolate, :periodic),
            form in (:nonconservative, :conservative)

        nx, ny = 24, 16
        halo = 3
        dt, dx, dy = 0.001, inv(nx), inv(ny)
        b1 = bk === :extrapolate ? ExtrapolateBC() : PeriodicBC()
        boundary = ntuple(_ -> b1, 4)

        u_s = serial_reference_2d(nx, ny, form, boundary; dt, dx, dy)

        topo = weno_cartesian_topology((nx, ny); comm, halo, periodic = bk === :periodic)
        owned = weno_owned_size(topo)
        offset = weno_global_offset(topo)

        c0 = zeros(owned .+ 2halo)
        vx = zeros(owned .+ 2halo)
        vy = zeros(owned .+ 2halo)
        for j in 1:owned[2], i in 1:owned[1]
            gi, gj = offset[1] + i, offset[2] + j
            c0[halo + i, halo + j] = gfield(gi, gj, nx, ny)
            vx[halo + i, halo + j] = gfield(gi, gj, nx, ny) + 0.4
            vy[halo + i, halo + j] = gfield(gi, gj, nx, ny) + 0.1
        end
        weno = WENOScheme(c0, topo; boundary, form, stag = false, multithreading = false)
        WENO_step!(c0, (; x = vx, y = vy), weno, dt, dx, dy)

        r1 = (halo + 1):(halo + owned[1])
        r2 = (halo + 1):(halo + owned[2])
        gr1 = (offset[1] + 1):(offset[1] + owned[1])
        gr2 = (offset[2] + 1):(offset[2] + owned[2])
        got = c0[r1, r2]
        expected = u_s[gr1, gr2]
        @test got == expected
    end

    # vx peaks at the global low corner and vy at the high corner, so on 2+
    # ranks the two maxima live on different ranks. Guards against a tuple
    # reduction that takes both LF speeds from one rank.
    @testset "2D conservative step: velocity components peak on different ranks" begin
        nx, ny = 24, 16
        halo = 3
        dt, dx, dy = 0.001, inv(nx), inv(ny)
        boundary = ntuple(_ -> ExtrapolateBC(), 4)
        vxg(gi, gj) = gi <= 2 && gj <= 2 ? 2.0 : 0.1
        vyg(gi, gj) = gi >= nx - 1 && gj >= ny - 1 ? 5.0 : 0.1

        u_s = [gfield(i, j, nx, ny) for i in 1:nx, j in 1:ny]
        weno_s = WENOScheme(copy(u_s); boundary, form = :conservative, stag = false, multithreading = false)
        vel_s = (x = [vxg(i, j) for i in 1:nx, j in 1:ny], y = [vyg(i, j) for i in 1:nx, j in 1:ny])
        WENO_step!(u_s, vel_s, weno_s, dt, dx, dy)

        topo = weno_cartesian_topology((nx, ny); comm, halo)
        owned = weno_owned_size(topo)
        offset = weno_global_offset(topo)
        c0 = zeros(owned .+ 2halo)
        vx = zeros(owned .+ 2halo)
        vy = zeros(owned .+ 2halo)
        for j in 1:owned[2], i in 1:owned[1]
            gi, gj = offset[1] + i, offset[2] + j
            c0[halo + i, halo + j] = gfield(gi, gj, nx, ny)
            vx[halo + i, halo + j] = vxg(gi, gj)
            vy[halo + i, halo + j] = vyg(gi, gj)
        end
        weno = WENOScheme(c0, topo; boundary, form = :conservative, stag = false, multithreading = false)
        WENO_step!(c0, (; x = vx, y = vy), weno, dt, dx, dy)

        r1 = (halo + 1):(halo + owned[1])
        r2 = (halo + 1):(halo + owned[2])
        @test c0[r1, r2] == u_s[(offset[1] + 1):(offset[1] + owned[1]), (offset[2] + 1):(offset[2] + owned[2])]
    end

    @testset "rank-asymmetric velocity, conservative, stag=false: matches serial" begin
        n = 40
        halo = 3
        dt, dx = 0.001, inv(n)
        boundary = (ExtrapolateBC(), ExtrapolateBC())

        # velocity nonzero only in the GLOBAL region owned by rank 0 in a
        # single-rank reference — but the REFERENCE here is itself computed
        # from the same global velocity field every rank agrees on, so
        # "rank-asymmetric" is about which RANK'S OWNED WINDOW holds the
        # nonzero region, not about different ranks disagreeing on the field.
        vglobal(gi) = gi <= n ÷ nprocs ? 2.0 : 0.0 # concentrated in the lowest rank's owned window

        u = [gfield(i, n) for i in 1:n]
        weno_s = WENOScheme(copy(u); boundary, form = :conservative, stag = false, multithreading = false)
        vel_s = (; x = [vglobal(i) for i in 1:n])
        u_s = copy(u)
        WENO_step!(u_s, vel_s, weno_s, dt, dx)

        topo = weno_cartesian_topology((n,); comm, halo)
        owned = weno_owned_size(topo)
        offset = weno_global_offset(topo)

        c0 = zeros(owned[1] + 2halo)
        v = zeros(owned[1] + 2halo)
        c0[(halo + 1):(halo + owned[1])] .= [gfield(offset[1] + i, n) for i in 1:owned[1]]
        v[(halo + 1):(halo + owned[1])] .= [vglobal(offset[1] + i) for i in 1:owned[1]]
        weno = WENOScheme(c0, topo; boundary, form = :conservative, stag = false, multithreading = false)
        WENO_step!(c0, (; x = v), weno, dt, dx)

        @test c0[(halo + 1):(halo + owned[1])] == u_s[(offset[1] + 1):(offset[1] + owned[1])]
    end

    @testset "multi-field tuple advection, rank-asymmetric velocity, conservative" begin
        n = 40
        halo = 3
        dt, dx = 0.001, inv(n)
        boundary = (ExtrapolateBC(), ExtrapolateBC())
        vglobal(gi) = gi <= n ÷ max(nprocs, 2) ? 1.5 : 0.0

        u1 = [gfield(i, n) for i in 1:n]
        u2 = [gfield(i, n) + 0.2 for i in 1:n]
        weno_s = WENOScheme(copy(u1); boundary, form = :conservative, stag = false, multithreading = false)
        vel_s = (; x = [vglobal(i) for i in 1:n])
        u1s, u2s = copy(u1), copy(u2)
        WENO_step!((u1s, u2s), vel_s, weno_s, dt, dx; u_min = (0.0, 0.0), u_max = (2.0, 2.0))

        topo = weno_cartesian_topology((n,); comm, halo)
        owned = weno_owned_size(topo)
        offset = weno_global_offset(topo)

        c1 = zeros(owned[1] + 2halo)
        c2 = zeros(owned[1] + 2halo)
        v = zeros(owned[1] + 2halo)
        c1[(halo + 1):(halo + owned[1])] .= [gfield(offset[1] + i, n) for i in 1:owned[1]]
        c2[(halo + 1):(halo + owned[1])] .= [gfield(offset[1] + i, n) + 0.2 for i in 1:owned[1]]
        v[(halo + 1):(halo + owned[1])] .= [vglobal(offset[1] + i) for i in 1:owned[1]]
        weno = WENOScheme(c1, topo; boundary, form = :conservative, stag = false, multithreading = false)
        WENO_step!((c1, c2), (; x = v), weno, dt, dx; u_min = (0.0, 0.0), u_max = (2.0, 2.0))

        r = (halo + 1):(halo + owned[1])
        gr = (offset[1] + 1):(offset[1] + owned[1])
        @test c1[r] == u1s[gr]
        @test c2[r] == u2s[gr]
    end

    @testset "lim_ZS = true agrees with serial (stag=false, nonconservative)" begin
        n = 40
        halo = 3
        dt, dx = 0.001, inv(n)
        boundary = (ExtrapolateBC(), ExtrapolateBC())

        u = [gfield(i, n) for i in 1:n]
        weno_s = WENOScheme(copy(u); boundary, form = :nonconservative, stag = false, lim_ZS = true, multithreading = false)
        vel_s = (; x = [gfield(i, n) + 0.5 for i in 1:n])
        u_s = copy(u)
        WENO_step!(u_s, vel_s, weno_s, dt, dx; u_min = 0.0, u_max = 2.0)

        topo = weno_cartesian_topology((n,); comm, halo)
        owned = weno_owned_size(topo)
        offset = weno_global_offset(topo)

        c0 = zeros(owned[1] + 2halo)
        v = zeros(owned[1] + 2halo)
        c0[(halo + 1):(halo + owned[1])] .= [gfield(offset[1] + i, n) for i in 1:owned[1]]
        v[(halo + 1):(halo + owned[1])] .= [gfield(offset[1] + i, n) + 0.5 for i in 1:owned[1]]
        weno = WENOScheme(c0, topo; boundary, form = :nonconservative, stag = false, lim_ZS = true, multithreading = false)
        WENO_step!(c0, (; x = v), weno, dt, dx; u_min = 0.0, u_max = 2.0)

        @test c0[(halo + 1):(halo + owned[1])] == u_s[(offset[1] + 1):(offset[1] + owned[1])]
    end
end

@testset "3D conservative parity over repeated steps, periodic y=$py" for py in (false, true)
    dims = (8nprocs, 6, 6)
    periodic = (true, py, true)
    topo = weno_cartesian_topology(dims; comm, dims = (nprocs, 1, 1), periodic)
    boundary = ntuple(f -> periodic[(f + 1) ÷ 2] ? PeriodicBC() : ExtrapolateBC(), 6)
    reference = [
        1 + 0.2sinpi(2i / dims[1]) * cospi(2j / dims[2]) + 0.1sinpi(2k / dims[3])
            for i in 1:dims[1], j in 1:dims[2], k in 1:dims[3]
    ]
    velocity = (x = reference .+ 0.5, y = reference .- 0.3, z = reference .+ 0.2)
    serial = WENOScheme(reference; boundary, form = :conservative, multithreading = false)
    local_state = allocate_weno_field(topo)
    ranges = weno_global_ranges(topo)
    owned_window(local_state, topo) .= view(reference, ranges...)
    local_velocity = map(velocity) do component
        a = allocate_weno_field(topo)
        owned_window(a, topo) .= view(component, ranges...)
        a
    end
    distributed = WENOScheme(local_state, topo; boundary, form = :conservative, multithreading = false)
    for _ in 1:3
        WENO_step!(reference, velocity, serial, 0.001, 0.1, 0.1, 0.1)
        WENO_step!(local_state, local_velocity, distributed, 0.001, 0.1, 0.1, 0.1)
        @test owned_window(local_state, topo) == view(reference, ranges...)
    end
end
