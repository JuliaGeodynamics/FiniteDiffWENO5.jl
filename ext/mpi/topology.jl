# MPI Cartesian topology: ranks, coordinates, neighbours, and reductions.

"""
    WENOCartesianTopology{N}

An MPI Cartesian decomposition of a global `(Nx, ...)` grid into `N`
dimensions. Built by [`weno_cartesian_topology`](@ref).
"""
struct WENOCartesianTopology{N} <: AbstractWENOTopology
    global_dims::NTuple{N, Int}
    halo::NTuple{N, Int}
    dims::NTuple{N, Int}          # process grid shape
    coords::NTuple{N, Int}        # this rank's coordinates in the process grid
    periodic::NTuple{N, Bool}
    comm::MPI.Comm                # the Cartesian communicator
    neighbors::NTuple{N, Tuple{Cint, Cint}} # (low, high) neighbour rank per axis, MPI.PROC_NULL if none
    rank::Int
end

"""
    weno_cartesian_topology(global_dims; comm = MPI.COMM_WORLD, halo = 3,
                             dims = nothing, periodic = false)

Build an MPI Cartesian topology over global cell counts. `dims` selects the
process grid; zero entries let MPI choose. `halo` and `periodic` accept scalars
or per-axis tuples. Each axis must divide evenly and own more cells than its
halo width to support staggered exchange.
"""
function weno_cartesian_topology(
        global_dims::NTuple{N, Int}; comm::MPI.Comm = MPI.COMM_WORLD,
        halo::Union{Int, NTuple{N, Int}} = 3,
        dims::Union{Nothing, NTuple{N, Int}} = nothing,
        periodic::Union{Bool, NTuple{N, Bool}} = false,
    ) where {N}
    MPI.Initialized() || MPI.Init()

    halo_t = halo isa Int ? ntuple(_ -> halo, N) : halo
    periodic_t = periodic isa Bool ? ntuple(_ -> periodic, N) : periodic

    nprocs = MPI.Comm_size(comm)
    dims_seed = dims === nothing ? zeros(Int, N) : collect(Int, dims)
    dims_t = NTuple{N, Int}(MPI.Dims_create(nprocs, dims_seed))

    all(d -> global_dims[d] % dims_t[d] == 0, 1:N) || throw(
        ArgumentError(
            "process grid $dims_t does not divide global dims $global_dims exactly on every axis"
        )
    )
    owned = ntuple(d -> global_dims[d] ÷ dims_t[d], N)
    # Staggered low-side sends need h+1 owned entries; otherwise they forward
    # a seam ghost that has not yet been received.
    all(d -> owned[d] >= halo_t[d] + 1, 1:N) || throw(
        ArgumentError(
            "owned cells per rank $owned must exceed the halo $halo_t by at least 1 on every axis " *
                "(process grid $dims_t over global dims $global_dims) — a subdomain this thin makes " *
                "a face-staggered exchange forward a not-yet-received ghost as if it were owned data"
        )
    )

    comm_cart = MPI.Cart_create(comm, collect(dims_t); periodic = collect(periodic_t), reorder = false)
    rank = MPI.Comm_rank(comm_cart)
    coords = NTuple{N, Int}(MPI.Cart_coords(comm_cart))

    neighbors = ntuple(N) do d
        low, high = MPI.Cart_shift(comm_cart, d - 1, 1)
        (Cint(low), Cint(high))
    end

    return WENOCartesianTopology{N}(global_dims, halo_t, dims_t, coords, periodic_t, comm_cart, neighbors, rank)
end

weno_ndims(::WENOCartesianTopology{N}) where {N} = N
weno_halo(topo::WENOCartesianTopology) = topo.halo

function weno_owned_size(topo::WENOCartesianTopology{N}; geometry::Symbol = :cell) where {N}
    per_axis = ntuple(d -> topo.global_dims[d] ÷ topo.dims[d], N)
    geometry !== :vertex && return per_axis
    # Vertex lattice: the low process rank on an axis owns `n+1` vertices,
    # every later rank owns `n` — disjoint ownership covering `Ncells+1`
    # global vertices exactly once per axis.
    return ntuple(d -> per_axis[d] + (topo.coords[d] == 0 ? 1 : 0), N)
end

function weno_global_size(topo::WENOCartesianTopology{N}; geometry::Symbol = :cell) where {N}
    geometry === :vertex && return topo.global_dims .+ 1
    return topo.global_dims
end

function weno_global_offset(topo::WENOCartesianTopology{N}; geometry::Symbol = :cell) where {N}
    per_axis = ntuple(d -> topo.global_dims[d] ÷ topo.dims[d], N)
    geometry !== :vertex && return ntuple(d -> topo.coords[d] * per_axis[d], N)
    # Rank 0 on an axis owns vertices 1:(n+1) (offset 0); every later rank
    # `c` owns the next `n` vertices starting after rank 0's extra one:
    # offset = c*n + 1.
    return ntuple(d -> topo.coords[d] == 0 ? 0 : topo.coords[d] * per_axis[d] + 1, N)
end

weno_periodic(topo::WENOCartesianTopology) = topo.periodic

"""`weno_physical_low(topo)[d]` is `dims[d] == 1 || (!periodic[d] && coord[d] == 0)`:
a single-rank axis is always physical on both faces — including when
periodic, where the wrap is supplied by `fill_physical_ghosts!`, not an
exchange — and a multi-rank periodic axis has no physical face at all."""
weno_physical_low(topo::WENOCartesianTopology{N}) where {N} =
    ntuple(d -> topo.dims[d] == 1 || (!topo.periodic[d] && topo.coords[d] == 0), N)

weno_physical_high(topo::WENOCartesianTopology{N}) where {N} =
    ntuple(d -> topo.dims[d] == 1 || (!topo.periodic[d] && topo.coords[d] == topo.dims[d] - 1), N)

weno_allreduce_max(value::Real, topo::WENOCartesianTopology) = MPI.Allreduce(value, max, topo.comm)
weno_allreduce_min(value::Real, topo::WENOCartesianTopology) = MPI.Allreduce(value, min, topo.comm)

# Reduce tuples through a Vector: MPI.jl would otherwise apply Julia's
# `max`/`min` to whole tuples, which compare lexicographically.
function weno_allreduce_max(value::NTuple{K, T}, topo::WENOCartesianTopology) where {K, T <: Real}
    return NTuple{K, T}(MPI.Allreduce(collect(value), max, topo.comm))
end
function weno_allreduce_min(value::NTuple{K, T}, topo::WENOCartesianTopology) where {K, T <: Real}
    return NTuple{K, T}(MPI.Allreduce(collect(value), min, topo.comm))
end

"""
    WENOScheme(c0::AbstractArray{T,N}, topo::WENOCartesianTopology; geometry = :cell,
               boundary, form, stag = false, lim_ZS = false,
               multithreading = true, upwind_mode = false)

Build a padded scalar scheme through the shared topology constructor.
"""
function FiniteDiffWENO5.WENOScheme(
        c0::AbstractArray{T, N}, topo::WENOCartesianTopology{N}; kwargs...,
    ) where {T, N}
    return FiniteDiffWENO5.build_topology_weno_scheme(c0, topo; kwargs...)
end

"""
    MultiphaseWENOScheme(phases, topo::WENOCartesianTopology; geometry = :cell,
                          boundary, stag = false, multithreading = true)

Build a padded multiphase scheme through the shared topology constructor.
"""
function FiniteDiffWENO5.MultiphaseWENOScheme(
        phases::Tuple{Vararg{Any, NP}}, topo::WENOCartesianTopology; kwargs...,
    ) where {NP}
    return FiniteDiffWENO5.build_topology_multiphase_scheme(phases, topo; kwargs...)
end
