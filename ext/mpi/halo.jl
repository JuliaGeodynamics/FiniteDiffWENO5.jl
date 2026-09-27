# Axis order lets later exchanges carry updated corner ghosts.

const _TAG_HIGH_GHOST = Cint(100)
const _TAG_LOW_GHOST = Cint(101)

"""
    WENOHaloBuffers{N, B}

Preallocated halo buffers and persistent MPI requests for one field.
Physical boundaries have `nothing` buffers on the corresponding side.
"""
struct WENOHaloBuffers{N, B}
    send_lo::NTuple{N, Union{Nothing, B}}
    recv_lo::NTuple{N, Union{Nothing, B}}
    send_hi::NTuple{N, Union{Nothing, B}}
    recv_hi::NTuple{N, Union{Nothing, B}}
    # Reuse each axis's persistent requests across exchanges.
    reqs::NTuple{N, MPI.RequestSet}
end

# Persistent requests retain buffer addresses; copying them would still target
# the original arrays. Build a new scheme to get independent buffers.
Base.deepcopy_internal(::WENOHaloBuffers, ::IdDict) = throw(
    ArgumentError(
        "WENOHaloBuffers cannot be deepcopy'd: its persistent MPI requests are bound " *
            "to specific send/recv buffer addresses, and a naive copy would leave both " *
            "the original and the copy exchanging into/out of the wrong arrays. Build a " *
            "new scheme from the same topology instead of deepcopy'ing an existing one."
    )
)

"""
    _build_halo_buffers(::Type{T}, topo, owned, halo, phys_lo, phys_hi, stagger, trailing)

Build per-axis buffers. A staggered axis uses one extra high-side entry;
`trailing` adds a phase dimension for fused exchanges.
"""
function _build_halo_buffers(
        ::Type{T}, topo::WENOCartesianTopology{N}, owned::NTuple{N, Int}, halo::NTuple{N, Int},
        phys_lo::NTuple{N, Bool}, phys_hi::NTuple{N, Bool},
        stagger::Union{Nothing, Int}, trailing::Union{Nothing, Int},
    ) where {T, N}
    full = ntuple(k -> owned[k] + 2halo[k] + (k === stagger ? 1 : 0), N)
    high_width(e) = e === stagger ? halo[e] + 1 : halo[e]
    function make(width, e)
        dims = ntuple(k -> k == e ? width : full[k], N)
        return trailing === nothing ? Array{T, N}(undef, dims) : Array{T, N + 1}(undef, dims..., trailing)
    end

    send_lo = ntuple(e -> phys_lo[e] ? nothing : make(high_width(e), e), N)
    recv_lo = ntuple(e -> phys_lo[e] ? nothing : make(halo[e], e), N)
    send_hi = ntuple(e -> phys_hi[e] ? nothing : make(halo[e], e), N)
    recv_hi = ntuple(e -> phys_hi[e] ? nothing : make(high_width(e), e), N)

    # Bind requests once to these buffers.
    reqs = ntuple(N) do e
        active = MPI.Request[]
        phys_lo[e] || push!(
            active,
            MPI.Send_init(send_lo[e], topo.comm; dest = topo.neighbors[e][1], tag = _TAG_HIGH_GHOST),
            MPI.Recv_init(recv_lo[e], topo.comm; source = topo.neighbors[e][1], tag = _TAG_LOW_GHOST),
        )
        phys_hi[e] || push!(
            active,
            MPI.Send_init(send_hi[e], topo.comm; dest = topo.neighbors[e][2], tag = _TAG_LOW_GHOST),
            MPI.Recv_init(recv_hi[e], topo.comm; source = topo.neighbors[e][2], tag = _TAG_HIGH_GHOST),
        )
        MPI.RequestSet(active)
    end

    B = trailing === nothing ? Array{T, N} : Array{T, N + 1}
    return WENOHaloBuffers{N, B}(send_lo, recv_lo, send_hi, recv_hi, reqs)
end

"""
    FiniteDiffWENO5.halo_buffers_for(topo::WENOCartesianTopology, extent, stag, T)

Build center and optional face-staggered buffer pools.
"""
function FiniteDiffWENO5.halo_buffers_for(
        topo::WENOCartesianTopology{N}, extent::FiniteDiffWENO5.PaddedExtent{N}, stag::Bool, ::Type{T},
    ) where {N, T}
    owned = weno_owned_size(topo; geometry = extent.geometry)
    halo = weno_halo(topo)
    phys_lo = weno_physical_low(topo)
    phys_hi = weno_physical_high(topo)
    center = _build_halo_buffers(T, topo, owned, halo, phys_lo, phys_hi, nothing, nothing)
    stag || return (; center)
    labels = (:x, :y, :z)[1:min(N, 3)]
    axes_bufs = ntuple(d -> _build_halo_buffers(T, topo, owned, halo, phys_lo, phys_hi, d, nothing), min(N, 3))
    return NamedTuple{(:center, labels...)}((center, axes_bufs...))
end

"""
    FiniteDiffWENO5.halo_buffers_for_multiphase(topo::WENOCartesianTopology, extent, stag, T, ::Val{NP})

Build buffer pools with a fused `NP`-phase center entry.
"""
function FiniteDiffWENO5.halo_buffers_for_multiphase(
        topo::WENOCartesianTopology{N}, extent::FiniteDiffWENO5.PaddedExtent{N}, stag::Bool, ::Type{T}, ::Val{NP},
    ) where {N, T, NP}
    owned = weno_owned_size(topo; geometry = extent.geometry)
    halo = weno_halo(topo)
    phys_lo = weno_physical_low(topo)
    phys_hi = weno_physical_high(topo)
    center = _build_halo_buffers(T, topo, owned, halo, phys_lo, phys_hi, nothing, NP)
    stag || return (; center)
    labels = (:x, :y, :z)[1:min(N, 3)]
    axes_bufs = ntuple(d -> _build_halo_buffers(T, topo, owned, halo, phys_lo, phys_hi, d, nothing), min(N, 3))
    return NamedTuple{(:center, labels...)}((center, axes_bufs...))
end

@inline function _axis_slice(sizes::NTuple{N, Int}, d::Int, lo::Int, hi::Int) where {N}
    return ntuple(k -> k == d ? (lo:hi) : (1:sizes[k]), N)
end

"""
    _axis_exchange!(buffers, field, topo, d, low_width, high_width, owned_lo, owned_hi)

Exchange axis `d`; send `high_width` entries low and `low_width` entries high.
"""
function _axis_exchange!(
        buffers::WENOHaloBuffers{N}, field::AbstractArray{T, N}, topo::WENOCartesianTopology, d::Int,
        low_width::Int, high_width::Int, owned_lo::Int, owned_hi::Int,
    ) where {T, N}
    phys_lo = weno_physical_low(topo)[d]
    phys_hi = weno_physical_high(topo)[d]
    (phys_lo && phys_hi) && return field

    sizes = size(field)

    phys_lo || copyto!(buffers.send_lo[d], view(field, _axis_slice(sizes, d, owned_lo, owned_lo + high_width - 1)...))
    phys_hi || copyto!(buffers.send_hi[d], view(field, _axis_slice(sizes, d, owned_hi - low_width + 1, owned_hi)...))

    reqs = buffers.reqs[d]
    MPI.Startall(reqs)
    MPI.Waitall(reqs)

    phys_lo ||
        (view(field, _axis_slice(sizes, d, owned_lo - low_width, owned_lo - 1)...) .= buffers.recv_lo[d])
    phys_hi ||
        (view(field, _axis_slice(sizes, d, owned_hi + 1, owned_hi + high_width)...) .= buffers.recv_hi[d])

    return field
end

"""
    _pack_into!(buf, fields::NTuple{NP}, ::Val{N}, d, lo, hi)

Pack field slices along the phase dimension.
"""
function _pack_into!(buf, fields::NTuple{NP}, V::Val{N}, d, lo, hi) where {NP, N}
    sizes = size(fields[1])
    slice = _axis_slice(sizes, d, lo, hi)
    for (q, f) in enumerate(fields)
        copyto!(selectdim(buf, N + 1, q), view(f, slice...))
    end
    return buf
end

"""
    _unpack_from!(fields::NTuple{NP}, buf, ::Val{N}, d, lo, hi)

Unpack the phase dimension into field slices.
"""
function _unpack_from!(fields::NTuple{NP}, buf, V::Val{N}, d, lo, hi) where {NP, N}
    sizes = size(fields[1])
    slice = _axis_slice(sizes, d, lo, hi)
    for (q, f) in enumerate(fields)
        view(f, slice...) .= selectdim(buf, N + 1, q)
    end
    return fields
end

"""
    _axis_exchange_fused!(buffers, fields::NTuple{NP}, topo, d, low_width, high_width, owned_lo, owned_hi)

Exchange all phases in one message per neighbour and axis.
"""
function _axis_exchange_fused!(
        buffers::WENOHaloBuffers{N}, fields::NTuple{NP, AbstractArray{T, N}}, topo::WENOCartesianTopology, d::Int,
        low_width::Int, high_width::Int, owned_lo::Int, owned_hi::Int,
    ) where {NP, T, N}
    phys_lo = weno_physical_low(topo)[d]
    phys_hi = weno_physical_high(topo)[d]
    (phys_lo && phys_hi) && return fields

    V = Val(N)

    phys_lo || _pack_into!(buffers.send_lo[d], fields, V, d, owned_lo, owned_lo + high_width - 1)
    phys_hi || _pack_into!(buffers.send_hi[d], fields, V, d, owned_hi - low_width + 1, owned_hi)

    reqs = buffers.reqs[d]
    MPI.Startall(reqs)
    MPI.Waitall(reqs)

    phys_lo ||
        _unpack_from!(fields, buffers.recv_lo[d], V, d, owned_lo - low_width, owned_lo - 1)
    phys_hi ||
        _unpack_from!(fields, buffers.recv_hi[d], V, d, owned_hi + 1, owned_hi + high_width)

    return fields
end

"""
    weno_exchange_halo!(fields::NTuple{NP}, topo::WENOCartesianTopology, buffers::WENOHaloBuffers; geometry = :cell, stagger = nothing)

Exchange equal-shaped phase arrays using preallocated buffers and persistent
requests. Field and buffer element types must match.
"""
function weno_exchange_halo!(
        fields::NTuple{NP, AbstractArray{T, N}}, topo::WENOCartesianTopology{N}, buffers::WENOHaloBuffers{N, B};
        geometry::Symbol = :cell, stagger::Union{Nothing, Int} = nothing,
    ) where {NP, T, N, B}
    T === eltype(B) || throw(
        ArgumentError(
            "weno_exchange_halo!: field element type $T does not match the scheme's " *
                "halo-buffer element type $(eltype(B)) — mixed-precision exchange is not supported"
        )
    )
    geometry === :vertex && stagger !== nothing && throw(
        ArgumentError("geometry = :vertex requires stagger = nothing")
    )
    owned = weno_owned_size(topo; geometry)
    halo = weno_halo(topo)
    for d in 1:N
        h = halo[d]
        owned_lo = h + 1
        owned_hi = h + owned[d]
        if d == stagger
            _axis_exchange_fused!(buffers, fields, topo, d, h, h + 1, owned_lo, owned_hi)
        else
            _axis_exchange_fused!(buffers, fields, topo, d, h, h, owned_lo, owned_hi)
        end
    end
    return fields
end

"""
    weno_exchange_halo!(field, topo::WENOCartesianTopology, buffers::WENOHaloBuffers; geometry = :cell, stagger = nothing)

Exchange cell, face-staggered, or vertex ghosts using preallocated buffers.
The staggered axis receives `h` low ghosts and `h+1` high ghosts. Field and
buffer element types must match.
"""
function weno_exchange_halo!(
        field::AbstractArray{T, N}, topo::WENOCartesianTopology{N}, buffers::WENOHaloBuffers{N, B};
        geometry::Symbol = :cell, stagger::Union{Nothing, Int} = nothing,
    ) where {T, N, B}
    T === eltype(B) || throw(
        ArgumentError(
            "weno_exchange_halo!: field element type $T does not match the scheme's " *
                "halo-buffer element type $(eltype(B)) — mixed-precision exchange is not supported"
        )
    )
    geometry === :vertex && stagger !== nothing && throw(
        ArgumentError("geometry = :vertex requires stagger = nothing")
    )
    owned = weno_owned_size(topo; geometry)
    halo = weno_halo(topo)
    for d in 1:N
        h = halo[d]
        owned_lo = h + 1
        owned_hi = h + owned[d]
        if d == stagger
            _axis_exchange!(buffers, field, topo, d, h, h + 1, owned_lo, owned_hi)
        else
            _axis_exchange!(buffers, field, topo, d, h, h, owned_lo, owned_hi)
        end
    end
    return field
end
