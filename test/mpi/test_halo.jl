using Test
using MPI
using FiniteDiffWENO5
using FiniteDiffWENO5: weno_cartesian_topology, weno_exchange_halo!, weno_ndims,
    weno_halo, weno_owned_size, weno_global_offset, weno_global_size,
    weno_physical_low, weno_physical_high, allocate_weno_field, owned_window,
    weno_global_ranges

const comm = MPI.COMM_WORLD
const nprocs = MPI.Comm_size(comm)
const rank = MPI.Comm_rank(comm)

# A value that is a pure function of GLOBAL coordinates: whichever rank's
# data ultimately reaches a given ghost cell (one hop or, at a corner, two),
# it must equal this function evaluated at that ghost's global coordinates —
# that is what "checked analytically" means here.
cellvalue(gi::Int) = 1000.0 * gi
cellvalue(gi::Int, gj::Int) = 1000.0 * gi + gj
cellvalue(gi::Int, gj::Int, gk::Int) = 1000.0 * gi + 10.0 * gj + gk

const SENTINEL = -1.0e9 # never a legitimate `cellvalue`, marks "untouched"

"""Fill `a`'s owned window with `cellvalue` at each cell's global coordinates."""
function fill_owned!(a, topo; geometry = :cell)
    offset = weno_global_offset(topo; geometry)
    ow = owned_window(a, topo; geometry)
    for I in CartesianIndices(ow)
        ow[I] = cellvalue((Tuple(I) .+ offset)...)
    end
    return a
end

"""
Check every ghost of a cell-centred/vertex-lattice exchange against
`cellvalue` at its global coordinates, wrapping through `global_size` when
`periodic[d]`; a ghost whose global coordinates fall outside the domain on a
non-periodic axis (a physical boundary ghost — untouched by exchange) must
still read `SENTINEL`.
"""
function check_cell_exchange(a, topo; geometry = :cell, periodic = ntuple(_ -> false, weno_ndims(topo)))
    N = weno_ndims(topo)
    halo = weno_halo(topo)
    owned = weno_owned_size(topo; geometry)
    offset = weno_global_offset(topo; geometry)
    gsize = weno_global_size(topo; geometry)
    lo = halo .+ 1
    hi = halo .+ owned
    phys_lo = weno_physical_low(topo)
    phys_hi = weno_physical_high(topo)

    for I in CartesianIndices(a)
        idx = Tuple(I)
        is_owned = all(d -> lo[d] <= idx[d] <= hi[d], 1:N)
        is_owned && continue

        # A ghost is only ever filled by exchange along an axis where THIS
        # rank has a real neighbour on the side the ghost extends into — not
        # merely wherever the axis happens to have several ranks overall (a
        # rank at the end of a non-periodic multi-rank axis has a physical
        # face on that end regardless of the OTHER end having a neighbour).
        touched = all(1:N) do d
            idx[d] < lo[d] && return !phys_lo[d]
            idx[d] > hi[d] && return !phys_hi[d]
            return true
        end
        if !touched
            @test a[I] == SENTINEL
            continue
        end

        g = ntuple(d -> idx[d] - halo[d] + offset[d], N)
        gwrapped = ntuple(d -> periodic[d] ? mod1(g[d], gsize[d]) : g[d], N)
        @test a[I] == cellvalue(gwrapped...)
    end
    return nothing
end

const CELL_2D_CASES = nprocs == 4 ?
    (((24, 16), nothing), ((24, 16), (4, 1))) : # the second case forces an interior rank with neighbours on both sides of axis 1
    (((24, 16), nothing),)

@testset "cell-centred exchange, 2D" for (global_dims, forced_dims) in CELL_2D_CASES
    N = 2
    halo = 3
    topo = weno_cartesian_topology(global_dims; comm, halo, dims = forced_dims)
    owned = weno_owned_size(topo)
    npad = owned .+ 2halo

    a = fill(SENTINEL, npad)
    fill_owned!(a, topo)
    weno_exchange_halo!(a, topo)
    check_cell_exchange(a, topo)

    if forced_dims !== nothing && nprocs == 4
        # rank 1 (0-indexed, dims=(4,1)) is interior on axis 1: neither face physical
        rank == 1 && @test !weno_physical_low(topo)[1] && !weno_physical_high(topo)[1]
    end
end

@testset "cell-centred exchange, 3D" begin
    N = 3
    halo = 3
    global_dims = (12, 8, 8)
    topo = weno_cartesian_topology(global_dims; comm, halo)
    owned = weno_owned_size(topo)
    npad = owned .+ 2halo

    a = fill(SENTINEL, npad)
    fill_owned!(a, topo)
    weno_exchange_halo!(a, topo)
    check_cell_exchange(a, topo)
end

@testset "cell-centred exchange, periodic 2D" begin
    N = 2
    halo = 3
    global_dims = (24, 16)
    topo = weno_cartesian_topology(global_dims; comm, halo, periodic = true)
    owned = weno_owned_size(topo)
    npad = owned .+ 2halo

    a = fill(SENTINEL, npad)
    fill_owned!(a, topo)
    weno_exchange_halo!(a, topo)
    check_cell_exchange(a, topo; periodic = (true, true))
end

@testset "an axis with one rank does no self-exchange" begin
    # A genuinely 1-rank *axis* — the communicator itself may still have
    # several ranks (the other axis carries them), so this is meaningful at
    # every process count this suite runs under.
    halo = 3
    global_dims = (24, 16)
    topo = weno_cartesian_topology(global_dims; comm, halo, dims = nprocs == 1 ? (1, 1) : (1, nprocs))
    @test weno_physical_low(topo)[1] && weno_physical_high(topo)[1] # axis 1 always single-rank here
    owned = weno_owned_size(topo)
    npad = owned .+ 2halo
    a = fill(SENTINEL, npad)
    fill_owned!(a, topo)
    weno_exchange_halo!(a, topo)
    # axis-1 ghosts are untouched (still sentinel); axis-2 ghosts, if any
    # neighbour exists, are filled.
    for I in CartesianIndices(a)
        i, j = Tuple(I)
        (halo < i <= halo + owned[1]) || (@test a[I] == SENTINEL)
    end
end

@testset "face-staggered exchange: every entry to n+2h+1, interior seam and physical" for stag_axis in (1, 2)
    halo = 3
    global_dims = (24, 16)
    topo = weno_cartesian_topology(global_dims; comm, halo)
    N = 2
    owned = weno_owned_size(topo)
    offset = weno_global_offset(topo)
    npad = ntuple(d -> d == stag_axis ? owned[d] + 2halo + 1 : owned[d] + 2halo, N)

    facevalue(gs) = 1000.0 * gs # depends only on the staggered axis's global face index

    a = fill(SENTINEL, npad)
    # Fix the tangential coordinate at a safely-owned index throughout: the
    # tangential axis is ALSO exchanged (symmetric h/h) by this same call,
    # but only at its own ghosts, never at this fixed owned index — so it
    # cannot disturb the stagger-axis check below.
    tang_axis = stag_axis == 1 ? 2 : 1
    tang_local = owned[tang_axis] ÷ 2 + 1
    tang_padded = halo + tang_local

    # Fill every OWNED face: this rank's low face through its high face —
    # the genuine extra physical face when the high side is physical, else
    # just its low..(low+owned-1) faces (the shared/ghost-received h+n+1'th
    # entry is intentionally left untouched here — it is the *other* rank's
    # data to send, not this rank's to fill).
    phys_hi = weno_physical_high(topo)[stag_axis]
    own_hi_local = phys_hi ? owned[stag_axis] + 1 : owned[stag_axis]
    for s_local in 1:own_hi_local
        s_padded = halo + s_local
        gs = offset[stag_axis] + s_local
        idx = stag_axis == 1 ? (s_padded, tang_padded) : (tang_padded, s_padded)
        a[idx...] = facevalue(gs)
    end

    weno_exchange_halo!(a, topo; stagger = stag_axis)

    lo = halo + 1
    hi_owned = halo + own_hi_local
    for s_padded in 1:npad[stag_axis]
        (lo <= s_padded <= hi_owned) && continue # this rank's own owned entries: already checked by construction
        gs = s_padded - halo + offset[stag_axis]
        idx = stag_axis == 1 ? (s_padded, tang_padded) : (tang_padded, s_padded)
        in_domain = 1 <= gs <= weno_global_size(topo)[stag_axis] + 1
        if in_domain
            @test a[idx...] == facevalue(gs)
        else
            @test a[idx...] == SENTINEL
        end
    end
end

@testset "face-staggered periodic wrap: global face N+1 duplicates face 1" begin
    halo = 3
    global_dims = (24,)
    topo = weno_cartesian_topology(global_dims; comm, halo, periodic = true)
    N = 1
    owned = weno_owned_size(topo)
    offset = weno_global_offset(topo)
    npad = owned[1] + 2halo + 1

    a = fill(SENTINEL, npad)
    for i in 1:owned[1] # only the primary n faces — never the duplicate
        a[halo + i] = 1000.0 * (offset[1] + i)
    end
    weno_exchange_halo!(a, topo; stagger = 1)

    # A single-rank periodic axis wraps through physical ghost filling.
    # Halo exchange handles periodic seams only when multiple ranks share it.
    n_global = global_dims[1]
    for i in 1:npad
        (halo < i <= halo + owned[1]) && continue
        if nprocs == 1
            @test a[i] == SENTINEL
            continue
        end
        g = i - halo + offset[1]
        gwrapped = mod1(g, n_global) # global face N+1 wraps to face 1
        @test a[i] == 1000.0 * gwrapped
    end
end

@testset "weno_exchange_halo!(fields::Tuple, ...) matches per-array exchange exactly" begin
    halo = 3
    global_dims = (16, 12)
    topo = weno_cartesian_topology(global_dims; comm, halo)
    owned = weno_owned_size(topo)
    npad = owned .+ 2halo

    a1 = fill(SENTINEL, npad)
    a2 = fill(SENTINEL, npad)
    fill_owned!(a1, topo)
    fill_owned!(a2, topo)
    b1 = copy(a1)
    b2 = copy(a2)

    weno_exchange_halo!(a1, topo)
    weno_exchange_halo!(a2, topo)
    weno_exchange_halo!((b1, b2), topo)

    @test a1 == b1
    @test a2 == b2
end

@testset "vertex-lattice: Ncells+1 global entries, each owned exactly once, exchange analytic" begin
    halo = 3
    global_dims = (12, 8)
    topo = weno_cartesian_topology(global_dims; comm, halo)
    N = 2

    owned_v = weno_owned_size(topo; geometry = :vertex)
    offset_v = weno_global_offset(topo; geometry = :vertex)
    global_v = weno_global_size(topo; geometry = :vertex)
    @test global_v == global_dims .+ 1

    # Per-axis ownership partitions 1:(Ncells+1) with no gap/overlap: gather
    # every rank's (offset, owned) pair along each axis and check the union.
    # Only ONE communicator rank per distinct coordinate along axis `d` may
    # contribute — every rank sharing that coordinate (differing only on
    # OTHER axes) reports the identical (offset, owned) pair, so summing
    # over every communicator rank would double- (or worse) count it; the
    # "representative" is the rank whose coordinate is 0 on every OTHER axis.
    for d in 1:N
        is_representative = all(k -> k == d || topo.coords[k] == 0, 1:N)
        payload = MPI.Allgather((is_representative, offset_v[d], owned_v[d]), comm)
        ranges = [((o + 1):(o + n)) for (rep, o, n) in payload if rep]
        total = sum(length, ranges)
        @test total == global_v[d]
        covered = reduce(union, ranges)
        @test covered == 1:global_v[d]
        # no overlap: total length equals the union's length
        @test length(covered) == total
    end

    npad = owned_v .+ 2halo
    a = fill(SENTINEL, npad)
    fill_owned!(a, topo; geometry = :vertex)
    weno_exchange_halo!(a, topo; geometry = :vertex)
    check_cell_exchange(a, topo; geometry = :vertex)
end

@testset "geometry = :vertex requires stagger = nothing" begin
    halo = 3
    topo = weno_cartesian_topology((12, 8); comm, halo)
    a = allocate_weno_field(topo; geometry = :vertex)
    @test_throws ArgumentError weno_exchange_halo!(a, topo; geometry = :vertex, stagger = 1)
end

@testset "a subdomain thinner than the halo fails with a clear message" begin
    if nprocs >= 4
        halo = 3
        # global 8 cells over 4 ranks along axis 1 → owned = 2 < halo = 3
        @test_throws ArgumentError weno_cartesian_topology((8, 8); comm, halo, dims = (4, 1))
    end
end

# Staggered low-side sends require at least halo + 1 owned entries.
@testset "owned == halo is rejected; owned == halo + 1 is accepted" begin
    if nprocs > 1
        halo = 3
        # global halo*nprocs cells, split evenly → owned == halo exactly
        @test_throws ArgumentError weno_cartesian_topology((halo * nprocs,); comm, halo)
        # global (halo+1)*nprocs cells, split evenly → owned == halo + 1
        topo = weno_cartesian_topology(((halo + 1) * nprocs,); comm, halo)
        @test weno_owned_size(topo) == (halo + 1,)
    end
end
