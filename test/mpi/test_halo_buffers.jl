using Test
using MPI
using FiniteDiffWENO5 # WENOScheme, weno_cartesian_topology, PeriodicBC, ExtrapolateBC are all exported
using FiniteDiffWENO5: weno_exchange_halo!

@testset "halo buffer pool shape and reuse" begin
    N = 2
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    halo_width = 3
    # axis 1 scales with rank count so owned == halo_width+1 on every rank
    # exactly (the repo's own convention, e.g. test_halo.jl:311); axis 2 stays
    # single-rank (dims[2] == 1) so it just needs owned >= halo_width+1.
    # axis 1's boundary below is periodic, so the topology's own axis 1 must
    # be periodic too (`resolve_boundary` rejects a periodicity mismatch
    # between a boundary face and the topology it's built against).
    topo = weno_cartesian_topology(
        ((halo_width + 1) * nprocs, halo_width + 1); halo = halo_width, dims = (nprocs, 1),
        periodic = (true, false),
    )
    halo = FiniteDiffWENO5.weno_halo(topo)
    owned = FiniteDiffWENO5.weno_owned_size(topo)

    scheme_collocated = WENOScheme(
        zeros(owned[1] + 2halo[1], owned[2] + 2halo[2]), topo;
        boundary = (PeriodicBC(), PeriodicBC(), ExtrapolateBC(), ExtrapolateBC()),
        form = :nonconservative, stag = false,
    )
    buffers_collocated = scheme_collocated.halo_buffers
    @test propertynames(buffers_collocated) == (:center,)

    scheme_staggered = WENOScheme(
        zeros(owned[1] + 2halo[1], owned[2] + 2halo[2]), topo;
        boundary = (PeriodicBC(), PeriodicBC(), ExtrapolateBC(), ExtrapolateBC()),
        form = :nonconservative, stag = true,
    )
    buffers_staggered = scheme_staggered.halo_buffers
    @test Set(propertynames(buffers_staggered)) == Set((:center, :x, :y))

    phys_lo = FiniteDiffWENO5.weno_physical_low(topo)
    phys_hi = FiniteDiffWENO5.weno_physical_high(topo)
    center = buffers_collocated.center
    for e in 1:N
        if phys_lo[e]
            @test center.send_lo[e] === nothing
            @test center.recv_lo[e] === nothing
        else
            @test center.send_lo[e] !== nothing
            @test size(center.send_lo[e], e) == halo[e]
        end
        if phys_hi[e]
            @test center.send_hi[e] === nothing
            @test center.recv_hi[e] === nothing
        else
            @test center.send_hi[e] !== nothing
            @test size(center.send_hi[e], e) == halo[e]
        end
    end

    # The staggered axis's high-side width is halo[e] + 1, not halo[e].
    bx = buffers_staggered.x
    if !phys_lo[1]
        @test size(bx.send_lo[1], 1) == halo[1] + 1
        @test size(bx.recv_hi[1], 1) == halo[1] + 1
    end
end

@testset "mismatched element type is rejected, not silently rounded" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    halo_width = 3
    topo = weno_cartesian_topology(
        ((halo_width + 1) * nprocs, halo_width + 1); halo = halo_width, dims = (nprocs, 1),
        periodic = (true, false),
    )
    halo = FiniteDiffWENO5.weno_halo(topo)
    owned = FiniteDiffWENO5.weno_owned_size(topo)

    scheme = WENOScheme(
        zeros(Float32, owned[1] + 2halo[1], owned[2] + 2halo[2]), topo;
        boundary = (PeriodicBC(), PeriodicBC(), ExtrapolateBC(), ExtrapolateBC()),
        form = :nonconservative, stag = false,
    )
    mismatched = zeros(Float64, owned[1] + 2halo[1], owned[2] + 2halo[2])
    @test_throws ArgumentError weno_exchange_halo!(mismatched, topo, scheme.halo_buffers.center)
end
