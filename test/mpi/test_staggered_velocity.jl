using Test
using MPI
using FiniteDiffWENO5
using FiniteDiffWENO5: weno_cartesian_topology, weno_owned_size, weno_global_offset,
    prepare_velocity!, AbstractWENOTopology

const comm = MPI.COMM_WORLD
const nprocs = MPI.Comm_size(comm)
const rank = MPI.Comm_rank(comm)

gface(gi, n) = sinpi(2 * gi / n)
gcenter_exact(gi, n) = sinpi(2 * (gi - 0.5) / n) # exact cell-centre value of sin(2πx)

@testset "distributed staggered velocity preparation" begin

    @testset "prepared vcenter matches serial over the owned window: 1D, 2, 4 ranks" begin
        n = 40
        halo = 3
        u = zeros(n) # only need a WENOScheme; the advected field itself is irrelevant here
        boundary = (ExtrapolateBC(), ExtrapolateBC())

        weno_s = WENOScheme(copy(u); boundary, form = :nonconservative, stag = true, multithreading = false)
        vface_s = [gface(i, n) for i in 0:n]
        vcenter_s = prepare_velocity!(weno_s, (; x = vface_s))

        topo = weno_cartesian_topology((n,); comm, halo)
        owned = weno_owned_size(topo)
        offset = weno_global_offset(topo)

        c0 = zeros(owned[1] + 2halo)
        weno_t = WENOScheme(c0, topo; boundary, form = :nonconservative, stag = true, multithreading = false)
        vface_t = zeros(owned[1] + 2halo + 1)
        vface_t[(halo + 1):(halo + owned[1] + 1)] .= [gface(offset[1] + i, n) for i in 0:owned[1]]
        vcenter_t = prepare_velocity!(weno_t, (; x = vface_t))

        got = vcenter_t.x[(halo + 1):(halo + owned[1])]
        expected = vcenter_s.x[(offset[1] + 1):(offset[1] + owned[1])]
        @test got == expected
    end

    @testset "prepared vcenter matches serial over the owned window: 2D" begin
        nx, ny = 24, 16
        halo = 3
        u = zeros(nx, ny)
        boundary = ntuple(_ -> ExtrapolateBC(), 4)

        weno_s = WENOScheme(copy(u); boundary, form = :nonconservative, stag = true, multithreading = false)
        vfx_s = [gface(i, nx) for i in 0:nx, j in 1:ny]
        vfy_s = [0.5 + 0.2cospi(2 * (j - 0.5) / ny) for i in 1:nx, j in 0:ny]
        vcenter_s = prepare_velocity!(weno_s, (x = vfx_s, y = vfy_s))

        topo = weno_cartesian_topology((nx, ny); comm, halo)
        owned = weno_owned_size(topo)
        offset = weno_global_offset(topo)

        c0 = zeros(owned .+ 2halo)
        weno_t = WENOScheme(c0, topo; boundary, form = :nonconservative, stag = true, multithreading = false)
        vfx_t = zeros(owned[1] + 2halo + 1, owned[2] + 2halo)
        vfy_t = zeros(owned[1] + 2halo, owned[2] + 2halo + 1)
        for j in 1:owned[2], i in 0:owned[1]
            vfx_t[halo + 1 + i, halo + j] = gface(offset[1] + i, nx)
        end
        for j in 0:owned[2], i in 1:owned[1]
            gj = offset[2] + j
            vfy_t[halo + i, halo + 1 + j] = 0.5 + 0.2cospi(2 * (gj - 0.5) / ny)
        end
        vcenter_t = prepare_velocity!(weno_t, (x = vfx_t, y = vfy_t))

        r1 = (halo + 1):(halo + owned[1])
        r2 = (halo + 1):(halo + owned[2])
        gr1 = (offset[1] + 1):(offset[1] + owned[1])
        gr2 = (offset[2] + 1):(offset[2] + owned[2])
        @test vcenter_t.x[r1, r2] == vcenter_s.x[gr1, gr2]
        @test vcenter_t.y[r1, r2] == vcenter_s.y[gr1, gr2]
    end

    @testset "fifth-order accuracy retained across a process seam" begin
        halo = 3
        global_errors = Float64[]
        for n in (16, 32, 64, 128)
            topo = weno_cartesian_topology((n,); comm, halo, periodic = true)
            owned = weno_owned_size(topo)
            offset = weno_global_offset(topo)

            c0 = zeros(owned[1] + 2halo)
            boundary = (PeriodicBC(), PeriodicBC())
            weno = WENOScheme(c0, topo; boundary, form = :nonconservative, stag = true, multithreading = false)

            vf = zeros(owned[1] + 2halo + 1)
            vf[(halo + 1):(halo + owned[1] + 1)] .= [gface(offset[1] + i, n) for i in 0:owned[1]]
            vcenter = prepare_velocity!(weno, (; x = vf))

            local_err = 0.0
            for i in 1:owned[1]
                gi = offset[1] + i
                local_err += abs(vcenter.x[halo + i] - gcenter_exact(gi, n))
            end
            total_err = MPI.Allreduce(local_err, +, comm)
            push!(global_errors, inv(float(n)) * total_err)
        end
        rates = log2.(global_errors[1:(end - 1)] ./ global_errors[2:end])
        @test all(>(4.5), rates)
    end

    @testset "globally periodic axis split across ranks matches the serial periodic result" begin
        n = 48
        halo = 3
        u = zeros(n)
        boundary = (PeriodicBC(), PeriodicBC())

        weno_s = WENOScheme(copy(u); boundary, form = :nonconservative, stag = true, multithreading = false)
        vface_s = [gface(i, n) for i in 0:(n - 1)] # serial periodic form: n samples
        vcenter_s = prepare_velocity!(weno_s, (; x = vface_s))

        topo = weno_cartesian_topology((n,); comm, halo, periodic = true)
        owned = weno_owned_size(topo)
        offset = weno_global_offset(topo)

        c0 = zeros(owned[1] + 2halo)
        weno_t = WENOScheme(c0, topo; boundary, form = :nonconservative, stag = true, multithreading = false)
        vface_t = zeros(owned[1] + 2halo + 1)
        vface_t[(halo + 1):(halo + owned[1])] .= [gface(offset[1] + i, n) for i in 0:(owned[1] - 1)] # duplicate high face left to fill_physical_ghosts!/exchange
        vcenter_t = prepare_velocity!(weno_t, (; x = vface_t))

        got = vcenter_t.x[(halo + 1):(halo + owned[1])]
        expected = vcenter_s.x[(offset[1] + 1):(offset[1] + owned[1])]
        @test got == expected
    end

    @testset "2-rank scalar WENO_step! with stag=true matches serial bit-for-bit ($form, $bk)" for form in (:nonconservative, :conservative),
            bk in (:extrapolate, :periodic)

        if nprocs in (2, 4)
            n = 40
            halo = 3
            dt, dx = 0.001, inv(n)
            boundary = bk === :extrapolate ? (ExtrapolateBC(), ExtrapolateBC()) : (PeriodicBC(), PeriodicBC())

            u = [1.0 + 0.3sinpi(2 * (i - 0.5) / n) + 0.15cospi(4 * (i - 0.5) / n) for i in 1:n]
            weno_s = WENOScheme(copy(u); boundary, form, stag = true, multithreading = false)
            vface_s = [gface(i, n) .+ 1.2 for i in 0:n]
            u_s = copy(u)
            WENO_step!(u_s, (; x = vface_s), weno_s, dt, dx)

            topo = weno_cartesian_topology((n,); comm, halo, periodic = bk === :periodic)
            owned = weno_owned_size(topo)
            offset = weno_global_offset(topo)

            c0 = zeros(owned[1] + 2halo)
            c0[(halo + 1):(halo + owned[1])] .= u[(offset[1] + 1):(offset[1] + owned[1])]
            weno = WENOScheme(c0, topo; boundary, form, stag = true, multithreading = false)
            vf = zeros(owned[1] + 2halo + 1)
            vf[(halo + 1):(halo + owned[1] + 1)] .= vface_s[(offset[1] + 1):(offset[1] + owned[1] + 1)]
            WENO_step!(c0, (; x = vf), weno, dt, dx)

            got = c0[(halo + 1):(halo + owned[1])]
            expected = u_s[(offset[1] + 1):(offset[1] + owned[1])]
            @test got == expected
        end
    end

    # A topology wrapper that counts every `weno_exchange_halo!` call and
    # forwards everything else to the real topology — a multiple-dispatch
    # counting shim, not a runtime function redefinition.
    struct CountingTopology{T} <: AbstractWENOTopology
        inner::T
        count::Base.RefValue{Int}
    end
    CountingTopology(inner) = CountingTopology(inner, Ref(0))
    FiniteDiffWENO5.weno_ndims(t::CountingTopology) = FiniteDiffWENO5.weno_ndims(t.inner)
    FiniteDiffWENO5.weno_halo(t::CountingTopology) = FiniteDiffWENO5.weno_halo(t.inner)
    FiniteDiffWENO5.weno_owned_size(t::CountingTopology; geometry = :cell) = FiniteDiffWENO5.weno_owned_size(t.inner; geometry)
    FiniteDiffWENO5.weno_global_size(t::CountingTopology; geometry = :cell) = FiniteDiffWENO5.weno_global_size(t.inner; geometry)
    FiniteDiffWENO5.weno_global_offset(t::CountingTopology; geometry = :cell) = FiniteDiffWENO5.weno_global_offset(t.inner; geometry)
    FiniteDiffWENO5.weno_periodic(t::CountingTopology) = FiniteDiffWENO5.weno_periodic(t.inner)
    FiniteDiffWENO5.weno_physical_low(t::CountingTopology) = FiniteDiffWENO5.weno_physical_low(t.inner)
    FiniteDiffWENO5.weno_physical_high(t::CountingTopology) = FiniteDiffWENO5.weno_physical_high(t.inner)
    function FiniteDiffWENO5.weno_exchange_halo!(field::AbstractArray, t::CountingTopology; kwargs...)
        t.count[] += 1
        return FiniteDiffWENO5.weno_exchange_halo!(field, t.inner; kwargs...)
    end
    FiniteDiffWENO5.weno_allreduce_max(v, t::CountingTopology) = FiniteDiffWENO5.weno_allreduce_max(v, t.inner)
    FiniteDiffWENO5.weno_allreduce_min(v, t::CountingTopology) = FiniteDiffWENO5.weno_allreduce_min(v, t.inner)

    @testset "prepare_velocity! runs once per WENO_step!, not per RK stage" begin
        n = 20
        halo = 3
        dt, dx = 0.001, inv(n)
        boundary = (ExtrapolateBC(), ExtrapolateBC())
        topo = CountingTopology(weno_cartesian_topology((n,); comm, halo))
        owned = weno_owned_size(topo.inner)

        c0 = zeros(owned[1] + 2halo)
        c0[(halo + 1):(halo + owned[1])] .= 1.0
        weno = FiniteDiffWENO5.build_topology_weno_scheme(
            c0, topo; boundary, form = :nonconservative, stag = true, multithreading = false,
        )
        vf = ones(owned[1] + 2halo + 1)

        WENO_step!(c0, (; x = vf), weno, dt, dx)

        # Exactly one exchange per exchange-table row for this call: row 1
        # (the single face-velocity component, since `prepare_velocity!` ran
        # once) + three `sync_stage!` calls (rows 3-5) = 4. If
        # `prepare_velocity!`'s exchange ran twice, this would be 5.
        @test topo.count[] == 4
    end
end
