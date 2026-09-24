using Test
using FiniteDiffWENO5
using FiniteDiffWENO5: boundary_faces, ProcessBC, AbstractWENOTopology, NoTopology,
    SerialTopology, resolve_boundary, owned_window,
    weno_ndims, weno_halo, weno_owned_size, weno_global_size, weno_global_offset,
    weno_periodic, weno_physical_low, weno_physical_high, weno_exchange_halo!,
    weno_allreduce_max, weno_allreduce_min

# Mock topology for boundary resolution and vertex ownership without MPI.
struct MockTopology{N} <: AbstractWENOTopology
    halo::NTuple{N, Int}
    owned::NTuple{N, Int}
    global_size::NTuple{N, Int}
    offset::NTuple{N, Int}
    periodic::NTuple{N, Bool}
    phys_lo::NTuple{N, Bool}
    phys_hi::NTuple{N, Bool}
    vertex_owned::NTuple{N, Int}
    vertex_global::NTuple{N, Int}
    vertex_offset::NTuple{N, Int}
end

FiniteDiffWENO5.weno_ndims(::MockTopology{N}) where {N} = N
FiniteDiffWENO5.weno_halo(t::MockTopology) = t.halo
FiniteDiffWENO5.weno_owned_size(t::MockTopology; geometry::Symbol = :cell) =
    geometry === :vertex ? t.vertex_owned : t.owned
FiniteDiffWENO5.weno_global_size(t::MockTopology; geometry::Symbol = :cell) =
    geometry === :vertex ? t.vertex_global : t.global_size
FiniteDiffWENO5.weno_global_offset(t::MockTopology; geometry::Symbol = :cell) =
    geometry === :vertex ? t.vertex_offset : t.offset
FiniteDiffWENO5.weno_periodic(t::MockTopology) = t.periodic
FiniteDiffWENO5.weno_physical_low(t::MockTopology) = t.phys_lo
FiniteDiffWENO5.weno_physical_high(t::MockTopology) = t.phys_hi
FiniteDiffWENO5.weno_exchange_halo!(field::AbstractArray, ::MockTopology; geometry::Symbol = :cell, stagger = nothing) = field
FiniteDiffWENO5.weno_allreduce_max(v, ::MockTopology) = v
FiniteDiffWENO5.weno_allreduce_min(v, ::MockTopology) = v

struct NoMethodsTopology end # implements nothing: every accessor must hit the fallback

# Boundary resolution accepts providers that implement the accessors without
# inheriting from `AbstractWENOTopology`.
struct ForeignTopology{N}
    periodic::NTuple{N, Bool}
    phys_lo::NTuple{N, Bool}
    phys_hi::NTuple{N, Bool}
end
FiniteDiffWENO5.weno_ndims(::ForeignTopology{N}) where {N} = N
FiniteDiffWENO5.weno_periodic(t::ForeignTopology) = t.periodic
FiniteDiffWENO5.weno_physical_low(t::ForeignTopology) = t.phys_lo
FiniteDiffWENO5.weno_physical_high(t::ForeignTopology) = t.phys_hi

# This provider also supports scheme construction, including halo-width checks.
struct ForeignSizedTopology{N}
    halo::NTuple{N, Int}
    owned::NTuple{N, Int}
end
FiniteDiffWENO5.weno_ndims(::ForeignSizedTopology{N}) where {N} = N
FiniteDiffWENO5.weno_halo(t::ForeignSizedTopology) = t.halo
FiniteDiffWENO5.weno_owned_size(t::ForeignSizedTopology; geometry::Symbol = :cell) = t.owned
FiniteDiffWENO5.weno_global_size(t::ForeignSizedTopology; geometry::Symbol = :cell) = t.owned
FiniteDiffWENO5.weno_periodic(t::ForeignSizedTopology{N}) where {N} = ntuple(_ -> false, N)
FiniteDiffWENO5.weno_physical_low(t::ForeignSizedTopology{N}) where {N} = ntuple(_ -> true, N)
FiniteDiffWENO5.weno_physical_high(t::ForeignSizedTopology{N}) where {N} = ntuple(_ -> true, N)

# A provider delegates construction to the shared topology builders.
FiniteDiffWENO5.WENOScheme(c0, topo::ForeignSizedTopology; kwargs...) =
    FiniteDiffWENO5.build_topology_weno_scheme(c0, topo; kwargs...)
FiniteDiffWENO5.MultiphaseWENOScheme(phases::Tuple, topo::ForeignSizedTopology; kwargs...) =
    FiniteDiffWENO5.build_topology_multiphase_scheme(phases, topo; kwargs...)

@testset "topology interface" begin

    @testset "every accessor has a throwing fallback naming the required extension" begin
        t = NoMethodsTopology()
        @test_throws ArgumentError weno_ndims(t)
        @test_throws ArgumentError weno_halo(t)
        @test_throws ArgumentError weno_owned_size(t)
        @test_throws ArgumentError weno_global_size(t)
        @test_throws ArgumentError weno_global_offset(t)
        @test_throws ArgumentError weno_periodic(t)
        @test_throws ArgumentError weno_physical_low(t)
        @test_throws ArgumentError weno_physical_high(t)
        @test_throws ArgumentError weno_exchange_halo!(zeros(3), t)
        @test_throws ArgumentError weno_allreduce_max(1.0, t)
        @test_throws ArgumentError weno_allreduce_min(1.0, t)
        # the message names the missing extension, not just "no method"
        try
            weno_ndims(t)
            @test false
        catch e
            @test e isa ArgumentError
            @test occursin("extension", e.msg)
            @test occursin("NoMethodsTopology", e.msg)
        end
        # the Tuple exchange method has a real default body (loops the
        # single-array method), so it throws TRANSITIVELY, not via its own
        # throwing fallback.
        @test_throws ArgumentError weno_exchange_halo!((zeros(3), zeros(3)), t)
    end

    @testset "NoTopology is not an AbstractWENOTopology; every accessor hits the fallback" begin
        t = NoTopology()
        @test !(t isa AbstractWENOTopology)
        @test_throws ArgumentError weno_ndims(t)
        @test_throws ArgumentError weno_halo(t)
        @test_throws ArgumentError weno_owned_size(t)
        @test_throws ArgumentError weno_periodic(t)
        @test_throws ArgumentError weno_physical_low(t)
        @test_throws ArgumentError weno_exchange_halo!(zeros(3), t)
        @test_throws ArgumentError weno_allreduce_max(1.0, t)
    end

    @testset "SerialTopology satisfies every accessor, N=$N, halo=$halo" for N in (1, 2, 3), halo in (0, 3)
        dims = ntuple(_ -> 8, N)
        topo = SerialTopology(dims; halo)
        @test weno_ndims(topo) == N
        @test weno_halo(topo) == ntuple(_ -> halo, N)
        @test weno_owned_size(topo) == dims
        @test weno_global_size(topo) == dims
        @test weno_global_offset(topo) == ntuple(_ -> 0, N)
        @test weno_periodic(topo) == ntuple(_ -> false, N)
        @test weno_physical_low(topo) == ntuple(_ -> true, N)
        @test weno_physical_high(topo) == ntuple(_ -> true, N)
        a = zeros(dims .+ 2halo)
        @test weno_exchange_halo!(a, topo) === a # no-op, single rank
        @test weno_allreduce_max(3.5, topo) == 3.5
        @test weno_allreduce_min(3.5, topo) == 3.5
        @test weno_allreduce_max((1.0, 2.0), topo) == (1.0, 2.0)

        # resolve_boundary under SerialTopology leaves all faces physical
        boundary = ntuple(_ -> ExtrapolateBC(), 2N)
        resolved = resolve_boundary(boundary, topo)
        @test all(b -> b isa ExtrapolateBC, resolved)

        boundary_periodic = ntuple(_ -> PeriodicBC(), 2N)
        topo_periodic = SerialTopology(dims; halo, periodic = true)
        resolved_periodic = resolve_boundary(boundary_periodic, topo_periodic)
        @test all(b -> b isa PeriodicBC, resolved_periodic) # single-rank periodic: BOTH faces stay physical
    end

    @testset "SerialTopology vertex geometry: global_cells + 1 entries" for N in (1, 2, 3)
        dims = ntuple(_ -> 6, N)
        topo = SerialTopology(dims)
        @test weno_owned_size(topo; geometry = :vertex) == dims .+ 1
        @test weno_global_size(topo; geometry = :vertex) == dims .+ 1
        @test weno_global_ranges(topo; geometry = :vertex) == ntuple(d -> 1:(dims[d] + 1), N)
    end

    @testset "resolve_boundary: geometry = :vertex requires stagger = nothing" begin
        topo = SerialTopology((6,))
        @test_throws ArgumentError allocate_weno_field(topo; geometry = :vertex, stagger = 1)
        @test_throws ArgumentError owned_window(zeros(12), topo; geometry = :vertex, stagger = 1)
    end

    @testset "resolve_boundary: every face x {low, interior, high} in 1D/2D/3D" begin
        for N in (1, 2, 3)
            for face_position in eachindex(1:(2N))
                d = (face_position + 1) ÷ 2
                is_low = isodd(face_position)
                for rank_position in (:low, :interior, :high)
                    phys_lo = ntuple(k -> k == d ? (rank_position === :low) : true, N)
                    phys_hi = ntuple(k -> k == d ? (rank_position === :high) : true, N)
                    topo = MockTopology(
                        ntuple(_ -> 3, N), ntuple(_ -> 4, N), ntuple(_ -> 4, N), ntuple(_ -> 0, N),
                        ntuple(_ -> false, N), phys_lo, phys_hi,
                        ntuple(_ -> 5, N), ntuple(_ -> 5, N), ntuple(_ -> 0, N),
                    )
                    boundary = ntuple(_ -> ExtrapolateBC(), 2N)
                    resolved = resolve_boundary(boundary, topo)

                    face_is_physical = is_low ? phys_lo[d] : phys_hi[d]
                    if face_is_physical
                        @test resolved[face_position] isa ExtrapolateBC
                    else
                        @test resolved[face_position] isa ProcessBC
                    end
                    # every OTHER face is untouched (this test's mock keeps
                    # every other axis fully physical)
                    for other in eachindex(1:(2N))
                        other == face_position && continue
                        od = (other + 1) ÷ 2
                        other_low = isodd(other)
                        od == d && continue # same-axis opposite face already covered by rank_position choice
                        other_physical = other_low ? phys_lo[od] : phys_hi[od]
                        @test (resolved[other] isa ExtrapolateBC) == other_physical
                    end
                end
            end
        end
    end

    @testset "resolve_boundary: periodic axis, single-rank keeps PeriodicBC, multi-rank yields ProcessBC" begin
        N = 2
        single_rank = MockTopology(
            (3, 3), (5, 5), (5, 5), (0, 0), (true, false), (true, true), (true, true),
            (6, 6), (6, 6), (0, 0),
        )
        boundary = (PeriodicBC(), PeriodicBC(), ExtrapolateBC(), ExtrapolateBC())
        resolved = resolve_boundary(boundary, single_rank)
        @test resolved[1] isa PeriodicBC && resolved[2] isa PeriodicBC

        multi_rank = MockTopology(
            (3, 3), (5, 5), (10, 5), (0, 0), (true, false), (false, true), (false, true),
            (6, 6), (11, 6), (0, 0),
        ) # an interior rank on axis 1: neither face physical
        resolved_mr = resolve_boundary(boundary, multi_rank)
        @test resolved_mr[1] isa ProcessBC && resolved_mr[2] isa ProcessBC
    end

    @testset "thin-subdomain guard applies to a foreign, non-subtyping topology too" begin
        halo = (3,)
        boundary = (ExtrapolateBC(), ExtrapolateBC())

        # owned == halo: fine for stag=false (only needs owned >= halo)...
        topo_eq = ForeignSizedTopology(halo, (3,))
        c0 = zeros(3 + 2 * 3)
        weno = WENOScheme(c0, topo_eq; boundary, form = :nonconservative, stag = false, multithreading = false)
        @test weno isa WENOScheme
        # ...but rejected for stag=true (needs owned >= halo + 1).
        @test_throws ArgumentError WENOScheme(c0, topo_eq; boundary, form = :nonconservative, stag = true, multithreading = false)

        p_eq = ntuple(_ -> zeros(3 + 2 * 3), 3)
        @test_throws ArgumentError MultiphaseWENOScheme(p_eq, topo_eq; boundary, stag = true, multithreading = false)

        # owned == halo + 1: accepted for stag=true too.
        topo_ok = ForeignSizedTopology(halo, (4,))
        c0_ok = zeros(4 + 2 * 3)
        weno_ok = WENOScheme(c0_ok, topo_ok; boundary, form = :nonconservative, stag = true, multithreading = false)
        @test weno_ok isa WENOScheme
        p_ok = ntuple(_ -> zeros(4 + 2 * 3), 3)
        scheme_ok = MultiphaseWENOScheme(p_ok, topo_ok; boundary, stag = true, multithreading = false)
        @test scheme_ok isa MultiphaseWENOScheme
    end

    @testset "resolve_boundary works on a topology that does not subtype AbstractWENOTopology" begin
        foreign = ForeignTopology((true, false), (false, true), (false, true))
        boundary = (PeriodicBC(), PeriodicBC(), ExtrapolateBC(), ExtrapolateBC())
        resolved = resolve_boundary(boundary, foreign)
        @test resolved[1] isa ProcessBC && resolved[2] isa ProcessBC # interior on axis 1
        @test resolved[3] isa ExtrapolateBC && resolved[4] isa ExtrapolateBC # physical on axis 2
    end

    @testset "boundary periodicity disagreeing with the topology's is rejected" begin
        topo_nonperiodic = SerialTopology((8,); periodic = false)
        @test_throws ArgumentError resolve_boundary((PeriodicBC(), PeriodicBC()), topo_nonperiodic)

        topo_periodic = SerialTopology((8,); periodic = true)
        @test_throws ArgumentError resolve_boundary((ExtrapolateBC(), ExtrapolateBC()), topo_periodic)

        # mismatched pairing (one face periodic, the other not) is rejected
        # regardless of the topology
        @test_throws ArgumentError resolve_boundary((PeriodicBC(), ExtrapolateBC()), topo_nonperiodic)
    end

    @testset "allocate_weno_field / owned_window / weno_global_ranges agree with the accessors" begin
        for N in (1, 2, 3)
            dims = ntuple(_ -> 10, N)
            halo = 3
            topo = SerialTopology(dims; halo)

            # cell-centred
            a = allocate_weno_field(topo)
            @test size(a) == dims .+ 2halo
            ow_cell = owned_window(a, topo)
            @test size(ow_cell) == dims
            # a `view`'s axes normalize to `Base.OneTo`; check it selects the
            # PHYSICAL entries by writing a marker into them in the parent
            # array and confirming the view sees exactly that.
            a .= 0.0
            a[ntuple(d -> (halo + 1):(halo + dims[d]), N)...] .= 1.0
            @test all(==(1.0), ow_cell)
            @test sum(a) == prod(dims) # nothing outside the owned window was touched

            # face-staggered along axis 1
            f = allocate_weno_field(topo; stagger = 1)
            expected = ntuple(d -> d == 1 ? dims[d] + 2halo + 1 : dims[d] + 2halo, N)
            @test size(f) == expected
            # owned window reaches one further along the staggered axis
            # (high side is physical for a single-rank topology)
            ow = owned_window(f, topo; stagger = 1)
            @test size(ow, 1) == dims[1] + 1
            for d in 2:N
                @test size(ow, d) == dims[d]
            end

            # vertex geometry
            v = allocate_weno_field(topo; geometry = :vertex)
            @test size(v) == (dims .+ 1) .+ 2halo
            @test size(owned_window(v, topo; geometry = :vertex)) == dims .+ 1

            @test weno_global_ranges(topo) == ntuple(d -> 1:dims[d], N)
        end
    end

    @testset "weno_global_ranges: two mock ranks partition a vertex axis with no gap or overlap" begin
        # rank 0 owns global vertices 1:(n+1), rank 1 owns the next n — the
        # low rank's extra vertex is the shared seam, owned exactly once.
        n = 5
        rank0 = MockTopology(
            (3,), (n,), (2n,), (0,), (false,), (true,), (false,),
            (n + 1,), (2n + 1,), (0,),
        )
        rank1 = MockTopology(
            (3,), (n,), (2n,), (n,), (false,), (false,), (true,),
            (n,), (2n + 1,), (n + 1,),
        )
        r0 = weno_global_ranges(rank0; geometry = :vertex)[1]
        r1 = weno_global_ranges(rank1; geometry = :vertex)[1]
        @test r0 == 1:(n + 1)
        @test r1 == (n + 2):(2n + 1)
        @test isempty(intersect(r0, r1))
        @test union(r0, r1) == 1:(2n + 1) # no gap
    end
end
