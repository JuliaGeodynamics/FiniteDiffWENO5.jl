# Exchange axes in order so later exchanges carry updated corner ghosts.

const _TAG_HIGH_GHOST = Cint(100) # data destined to fill the receiver's HIGH ghost
const _TAG_LOW_GHOST = Cint(101)  # data destined to fill the receiver's LOW ghost

@inline function _axis_slice(::Val{N}, d, lo, hi) where {N}
    return ntuple(k -> k == d ? (lo:hi) : Colon(), N)
end

"""
    _axis_exchange!(field, topo, d, low_width, high_width, owned_lo, owned_hi)

Exchange axis `d` with low and high neighbours. Send `high_width` owned
entries low and `low_width` entries high; receive the matching ghost widths.
Physical boundaries are filled separately.
"""
function _axis_exchange!(
        field::AbstractArray{T, N}, topo::WENOCartesianTopology, d::Int,
        low_width::Int, high_width::Int, owned_lo::Int, owned_hi::Int,
    ) where {T, N}
    phys_lo = weno_physical_low(topo)[d]
    phys_hi = weno_physical_high(topo)[d]
    (phys_lo && phys_hi) && return field # single rank on this axis: nothing to do

    low_rank, high_rank = topo.neighbors[d]
    comm = topo.comm
    V = Val(N)

    reqs = MPI.Request[]
    send_to_low = send_to_high = nothing
    recv_from_low = recv_from_high = nothing

    if !phys_lo
        send_to_low = collect(view(field, _axis_slice(V, d, owned_lo, owned_lo + high_width - 1)...))
        push!(reqs, MPI.Isend(send_to_low, low_rank, _TAG_HIGH_GHOST, comm))
        recv_from_low = Array{T, N}(undef, ntuple(k -> k == d ? low_width : size(field, k), N))
        push!(reqs, MPI.Irecv!(recv_from_low, low_rank, _TAG_LOW_GHOST, comm))
    end
    if !phys_hi
        send_to_high = collect(view(field, _axis_slice(V, d, owned_hi - low_width + 1, owned_hi)...))
        push!(reqs, MPI.Isend(send_to_high, high_rank, _TAG_LOW_GHOST, comm))
        recv_from_high = Array{T, N}(undef, ntuple(k -> k == d ? high_width : size(field, k), N))
        push!(reqs, MPI.Irecv!(recv_from_high, high_rank, _TAG_HIGH_GHOST, comm))
    end

    isempty(reqs) || MPI.Waitall(reqs)

    recv_from_low === nothing ||
        (view(field, _axis_slice(V, d, owned_lo - low_width, owned_lo - 1)...) .= recv_from_low)
    recv_from_high === nothing ||
        (view(field, _axis_slice(V, d, owned_hi + 1, owned_hi + high_width)...) .= recv_from_high)

    return field
end

"""
    _pack_slice(fields::NTuple{NP}, ::Val{N}, d, lo, hi)

Stack matching field slices along a phase dimension for one MPI message.
"""
function _pack_slice(fields::NTuple{NP}, V::Val{N}, d, lo, hi) where {NP, N}
    slice = _axis_slice(V, d, lo, hi)
    return cat((view(f, slice...) for f in fields)...; dims = N + 1)
end

"""
    _unpack_slice!(fields::NTuple{NP}, buf, ::Val{N}, d, lo, hi)

Scatter a packed buffer's phase dimension into field slices.
"""
function _unpack_slice!(fields::NTuple{NP}, buf, V::Val{N}, d, lo, hi) where {NP, N}
    slice = _axis_slice(V, d, lo, hi)
    for (q, f) in enumerate(fields)
        view(f, slice...) .= selectdim(buf, N + 1, q)
    end
    return fields
end

"""
    _axis_exchange_fused!(fields::NTuple{NP}, topo, d, low_width, high_width, owned_lo, owned_hi)

Exchange all phases together with one message per neighbour and axis.
"""
function _axis_exchange_fused!(
        fields::NTuple{NP, AbstractArray{T, N}}, topo::WENOCartesianTopology, d::Int,
        low_width::Int, high_width::Int, owned_lo::Int, owned_hi::Int,
    ) where {NP, T, N}
    phys_lo = weno_physical_low(topo)[d]
    phys_hi = weno_physical_high(topo)[d]
    (phys_lo && phys_hi) && return fields # single rank on this axis: nothing to do

    low_rank, high_rank = topo.neighbors[d]
    comm = topo.comm
    V = Val(N)

    reqs = MPI.Request[]
    recv_from_low = recv_from_high = nothing

    if !phys_lo
        send_to_low = collect(_pack_slice(fields, V, d, owned_lo, owned_lo + high_width - 1))
        push!(reqs, MPI.Isend(send_to_low, low_rank, _TAG_HIGH_GHOST, comm))
        recv_from_low = Array{T, N + 1}(
            undef, ntuple(k -> k == d ? low_width : size(fields[1], k), N)..., NP
        )
        push!(reqs, MPI.Irecv!(recv_from_low, low_rank, _TAG_LOW_GHOST, comm))
    end
    if !phys_hi
        send_to_high = collect(_pack_slice(fields, V, d, owned_hi - low_width + 1, owned_hi))
        push!(reqs, MPI.Isend(send_to_high, high_rank, _TAG_LOW_GHOST, comm))
        recv_from_high = Array{T, N + 1}(
            undef, ntuple(k -> k == d ? high_width : size(fields[1], k), N)..., NP
        )
        push!(reqs, MPI.Irecv!(recv_from_high, high_rank, _TAG_HIGH_GHOST, comm))
    end

    isempty(reqs) || MPI.Waitall(reqs)

    recv_from_low === nothing ||
        _unpack_slice!(fields, recv_from_low, V, d, owned_lo - low_width, owned_lo - 1)
    recv_from_high === nothing ||
        _unpack_slice!(fields, recv_from_high, V, d, owned_hi + 1, owned_hi + high_width)

    return fields
end

"""
    weno_exchange_halo!(fields::NTuple{NP}, topo::WENOCartesianTopology; geometry = :cell, stagger = nothing)

Exchange equal-shaped phase arrays together, one message per side and axis.
"""
function weno_exchange_halo!(
        fields::NTuple{NP, AbstractArray{T, N}}, topo::WENOCartesianTopology{N};
        geometry::Symbol = :cell, stagger::Union{Nothing, Int} = nothing,
    ) where {NP, T, N}
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
            _axis_exchange_fused!(fields, topo, d, h, h + 1, owned_lo, owned_hi)
        else
            _axis_exchange_fused!(fields, topo, d, h, h, owned_lo, owned_hi)
        end
    end
    return fields
end

"""
    weno_exchange_halo!(field, topo::WENOCartesianTopology; geometry = :cell, stagger = nothing)

Exchange cell, face-staggered, or vertex ghosts. The staggered axis receives
`h` low ghosts and `h+1` high ghosts: its extra high face belongs to the next
rank unless it is a physical boundary.
"""
function weno_exchange_halo!(
        field::AbstractArray{T, N}, topo::WENOCartesianTopology{N};
        geometry::Symbol = :cell, stagger::Union{Nothing, Int} = nothing,
    ) where {T, N}
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
            _axis_exchange!(field, topo, d, h, h + 1, owned_lo, owned_hi)
        else
            _axis_exchange!(field, topo, d, h, h, owned_lo, owned_hi)
        end
    end
    return field
end
