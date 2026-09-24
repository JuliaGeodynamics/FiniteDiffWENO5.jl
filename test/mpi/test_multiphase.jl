using Test
using MPI
using FiniteDiffWENO5
using FiniteDiffWENO5: weno_cartesian_topology, weno_owned_size, weno_global_offset

const comm = MPI.COMM_WORLD
const nprocs = MPI.Comm_size(comm)
const rank = MPI.Comm_rank(comm)

# deterministic three-phase composition of GLOBAL coordinates, summing to one
# everywhere by construction (third phase is 1 minus the other two)
function mp_phases(n)
    p1 = [0.3 + 0.15sinpi(2 * (i - 0.5) / n) for i in 1:n]
    p2 = [0.3 + 0.1cospi(4 * (i - 0.5) / n) for i in 1:n]
    return (p1, p2, [1 - p1[i] - p2[i] for i in 1:n])
end

function mp_phases(nx, ny)
    p1 = [0.3 + 0.1sinpi(2 * (i - 0.5) / nx) * cospi(2 * (j - 0.5) / ny) for i in 1:nx, j in 1:ny]
    p2 = [0.3 + 0.1cospi(2 * (i - 0.5) / nx) for i in 1:nx, j in 1:ny]
    return (p1, p2, [1 - p1[i, j] - p2[i, j] for i in 1:nx, j in 1:ny])
end

mp_face(gi, n) = 1.0 + 0.4sinpi(2 * gi / n)

periodic1D() = (PeriodicBC(), PeriodicBC())
periodic2D() = ntuple(_ -> PeriodicBC(), 4)
extrapolate1D() = (ExtrapolateBC(), ExtrapolateBC())
extrapolate2D() = ntuple(_ -> ExtrapolateBC(), 4)

function serial_reference_1d(n, boundary; dt, dx)
    p = mp_phases(n)
    v = (; x = [mp_face(i, n) for i in 0:n])
    scheme = MultiphaseWENOScheme(p; boundary, stag = true, multithreading = false)
    WENO_step!(p, v, scheme, dt, dx)
    return p
end

function serial_reference_2d(nx, ny, boundary; dt, dx, dy)
    p = mp_phases(nx, ny)
    v = (
        x = [mp_face(i, nx) for i in 0:nx, j in 1:ny],
        y = [0.5 + 0.2cospi(2 * (j - 0.5) / ny) for i in 1:nx, j in 0:ny],
    )
    scheme = MultiphaseWENOScheme(p; boundary, stag = true, multithreading = false)
    WENO_step!(p, v, scheme, dt, dx, dy)
    return p
end

@testset "distributed multiphase WENO_step!" begin

    @testset "1D bit-for-bit: $bk" for bk in (:extrapolate, :periodic)
        n = 40
        halo = 3
        dt, dx = 0.001, inv(n)
        boundary = bk === :extrapolate ? extrapolate1D() : periodic1D()

        p_s = serial_reference_1d(n, boundary; dt, dx)

        topo = weno_cartesian_topology((n,); comm, halo, periodic = bk === :periodic)
        owned = weno_owned_size(topo)
        offset = weno_global_offset(topo)

        all_p = mp_phases(n)
        c = ntuple(3) do q
            arr = zeros(owned[1] + 2halo)
            arr[(halo + 1):(halo + owned[1])] .= all_p[q][(offset[1] + 1):(offset[1] + owned[1])]
            arr
        end
        v = zeros(owned[1] + 2halo + 1)
        v[(halo + 1):(halo + owned[1] + 1)] .= [mp_face(offset[1] + i, n) for i in 0:owned[1]]

        scheme = MultiphaseWENOScheme(c, topo; boundary, stag = true, multithreading = false)
        WENO_step!(c, (; x = v), scheme, dt, dx)

        r = (halo + 1):(halo + owned[1])
        gr = (offset[1] + 1):(offset[1] + owned[1])
        for q in 1:3
            got = c[q][r]
            expected = p_s[q][gr]
            @test got == expected
        end
    end

    @testset "2D bit-for-bit: $bk" for bk in (:extrapolate, :periodic)
        nx, ny = 24, 16
        halo = 3
        dt, dx, dy = 0.001, inv(nx), inv(ny)
        boundary = bk === :extrapolate ? extrapolate2D() : periodic2D()

        p_s = serial_reference_2d(nx, ny, boundary; dt, dx, dy)

        topo = weno_cartesian_topology((nx, ny); comm, halo, periodic = bk === :periodic)
        owned = weno_owned_size(topo)
        offset = weno_global_offset(topo)

        all_p = mp_phases(nx, ny)
        c = ntuple(3) do q
            arr = zeros(owned .+ 2halo)
            for j in 1:owned[2], i in 1:owned[1]
                arr[halo + i, halo + j] = all_p[q][offset[1] + i, offset[2] + j]
            end
            arr
        end
        vx = zeros(owned[1] + 2halo + 1, owned[2] + 2halo)
        vy = zeros(owned[1] + 2halo, owned[2] + 2halo + 1)
        for j in 1:owned[2], i in 0:owned[1]
            vx[halo + 1 + i, halo + j] = mp_face(offset[1] + i, nx)
        end
        for j in 0:owned[2], i in 1:owned[1]
            gj = offset[2] + j
            vy[halo + i, halo + 1 + j] = 0.5 + 0.2cospi(2 * (gj - 0.5) / ny)
        end

        scheme = MultiphaseWENOScheme(c, topo; boundary, stag = true, multithreading = false)
        WENO_step!(c, (x = vx, y = vy), scheme, dt, dx, dy)

        r1 = (halo + 1):(halo + owned[1])
        r2 = (halo + 1):(halo + owned[2])
        gr1 = (offset[1] + 1):(offset[1] + owned[1])
        gr2 = (offset[2] + 1):(offset[2] + owned[2])
        for q in 1:3
            got = c[q][r1, r2]
            expected = p_s[q][gr1, gr2]
            @test got == expected
        end
    end

    @testset "sum-to-one and simplex bounds hold at seams after many steps" begin
        n = 60
        halo = 3
        dt, dx = 0.2 * inv(n), inv(n)
        boundary = periodic1D()

        topo = weno_cartesian_topology((n,); comm, halo, periodic = true)
        owned = weno_owned_size(topo)
        offset = weno_global_offset(topo)

        all_p = mp_phases(n)
        c = ntuple(3) do q
            arr = zeros(owned[1] + 2halo)
            arr[(halo + 1):(halo + owned[1])] .= all_p[q][(offset[1] + 1):(offset[1] + owned[1])]
            arr
        end
        v = zeros(owned[1] + 2halo + 1)
        v[(halo + 1):(halo + owned[1] + 1)] .= 1.0

        scheme = MultiphaseWENOScheme(c, topo; boundary, stag = true, multithreading = false)
        for _ in 1:100
            WENO_step!(c, (; x = v), scheme, dt, dx)
        end

        r = (halo + 1):(halo + owned[1])
        sums = c[1][r] .+ c[2][r] .+ c[3][r]
        local_max_sum_err = maximum(abs, sums .- 1)
        global_max_sum_err = MPI.Allreduce(local_max_sum_err, max, comm)
        @test global_max_sum_err < 1024eps(Float64)

        local_min = minimum(q -> minimum(q[r]), c)
        local_max = maximum(q -> maximum(q[r]), c)
        global_min = MPI.Allreduce(local_min, min, comm)
        global_max = MPI.Allreduce(local_max, max, comm)
        @test global_min >= -64eps(Float64)
        @test global_max <= 1 + 64eps(Float64)
    end

    @testset "fifth-order accuracy retained across a process seam" begin
        halo = 3
        global_errors = Float64[]
        for n in (32, 64, 128, 256)
            topo = weno_cartesian_topology((n,); comm, halo, periodic = true)
            owned = weno_owned_size(topo)
            offset = weno_global_offset(topo)

            smooth(gi) = 0.35 + 0.1sinpi(2 * (gi - 0.5) / n)
            exact1(gi) = smooth(gi)
            exact3(gi) = 1 - 2smooth(gi)

            c1 = zeros(owned[1] + 2halo)
            c2 = zeros(owned[1] + 2halo)
            c3 = zeros(owned[1] + 2halo)
            for i in 1:owned[1]
                gi = offset[1] + i
                c1[halo + i] = smooth(gi)
                c2[halo + i] = smooth(gi)
                c3[halo + i] = 1 - c1[halo + i] - c2[halo + i]
            end
            v = zeros(owned[1] + 2halo + 1)
            v[(halo + 1):(halo + owned[1] + 1)] .= 1.0

            scheme = MultiphaseWENOScheme(
                (c1, c2, c3), topo; boundary = periodic1D(), stag = true, multithreading = false
            )
            nt = ceil(Int, 1 / (0.4 * inv(n)^(5 / 3)))
            dt = 1 / nt
            for _ in 1:nt
                WENO_step!((c1, c2, c3), (; x = v), scheme, dt, inv(n))
            end

            local_err = 0.0
            for i in 1:owned[1]
                gi = offset[1] + i
                local_err += abs(c1[halo + i] - exact1(gi)) + abs(c3[halo + i] - exact3(gi))
            end
            total_err = MPI.Allreduce(local_err, +, comm)
            push!(global_errors, total_err / (2n))
        end
        rates = log2.(global_errors[1:(end - 1)] ./ global_errors[2:end])
        @test all(>(4.0), rates)
    end

    @testset "inflow physical faces install, ProcessBC no-op, full-step parity with serial" begin
        n = 40
        halo = 3
        dt, dx = 0.001, inv(n)
        inflow = (PrescribedInflowBC((0.6, 0.3, 0.1)), ExtrapolateBC())

        p_s = mp_phases(n)
        v_s = (; x = fill(1.0, n + 1))
        scheme_s = MultiphaseWENOScheme(p_s; boundary = inflow, stag = true, multithreading = false)
        WENO_step!(p_s, v_s, scheme_s, dt, dx)

        topo = weno_cartesian_topology((n,); comm, halo)
        owned = weno_owned_size(topo)
        offset = weno_global_offset(topo)

        all_p = mp_phases(n)
        c = ntuple(3) do q
            arr = zeros(owned[1] + 2halo)
            arr[(halo + 1):(halo + owned[1])] .= all_p[q][(offset[1] + 1):(offset[1] + owned[1])]
            arr
        end
        v = zeros(owned[1] + 2halo + 1)
        v[(halo + 1):(halo + owned[1] + 1)] .= 1.0

        scheme = MultiphaseWENOScheme(c, topo; boundary = inflow, stag = true, multithreading = false)
        # a ProcessBC face (any internal seam) must never overwrite ghosts with an
        # inflow/extrapolate value — resolve_boundary already installs it, this
        # just checks the resulting run matches the serial oracle over the owned window
        WENO_step!(c, (; x = v), scheme, dt, dx)

        r = (halo + 1):(halo + owned[1])
        gr = (offset[1] + 1):(offset[1] + owned[1])
        for q in 1:3
            @test c[q][r] == p_s[q][gr]
        end
    end
end
