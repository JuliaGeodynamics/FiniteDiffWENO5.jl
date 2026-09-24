@inline function left_index(i, d, nx, ::Val{0})
    # Dirichlet (clamped to domain)
    return clamp(i - d, 1, nx)
end

@inline function left_index(i, d, nx, ::Val{1})
    # Neumann (mirror the boundary value)
    return max(i - d, 1)
end

@inline function left_index(i, d, nx, ::Val{2})
    # Periodic (wrap around)
    return mod1(i - d, nx)
end

@inline function right_index(i, d, nx, ::Val{0})
    return clamp(i + d, 1, nx)   # Dirichlet
end

@inline function right_index(i, d, nx, ::Val{1})
    return min(i + d, nx)        # Neumann
end

@inline function right_index(i, d, nx, ::Val{2})
    return mod1(i + d, nx)       # Periodic
end

# runtime-boundary variants: used inside `@kernel` functions, where the boundary
# condition is a plain Int read from a kernel argument rather than known at compile
# time (so it can't be turned into a `Val` without triggering dynamic dispatch on
# the GPU). The branching happens once per index instead of being duplicated at
# every call site.
@inline function left_index(i, d, nx, b::Integer)
    if b == 0
        return clamp(i - d, 1, nx)   # Dirichlet
    elseif b == 1
        return max(i - d, 1)         # Neumann
    elseif b == 3
        return max(i - d, 1)         # ProcessBC (interior seam): clamp, like Neumann
    else
        return mod1(i - d, nx)       # Periodic
    end
end

@inline function right_index(i, d, nx, b::Integer)
    if b == 0
        return clamp(i + d, 1, nx)   # Dirichlet
    elseif b == 1
        return min(i + d, nx)        # Neumann
    elseif b == 3
        return min(i + d, nx)        # ProcessBC (interior seam): clamp, like Neumann
    else
        return mod1(i + d, nx)       # Periodic
    end
end

"""
    PaddedExtent{N}

Bookkeeping for how a `WENOScheme`/`MultiphaseWENOScheme` buffer's *allocated*
extent relates to its *physical* (owned) extent.

- `owned`: physical (owned) cell count per axis.
- `pad`: halo width per axis (0 for every axis on an unpadded/default scheme).
- `global_size`: global cell count per axis — equals `owned` outside a
  distributed context; recorded so the ENO5-vs-linear interpolation choice can
  key off the true global extent instead of the padded allocated one.
- `global_periodic`: the *serial* (physical) periodicity per axis — distinct
  from a scheme's `vperiodic`, which is forced `false` on every padded axis.
- `geometry`: `:cell` (cell lattice) or `:vertex` (vertex lattice).

Every existing serial construction gets the default: `owned == size(c0)`,
`pad = 0` on every axis, `global_size == owned`, `geometry = :cell`.
"""
struct PaddedExtent{N}
    owned::NTuple{N, Int}
    pad::NTuple{N, Int}
    global_size::NTuple{N, Int}
    global_periodic::NTuple{N, Bool}
    geometry::Symbol
end

"""Default extent for an unpadded scheme: no padding, physical extent equals
the allocated extent, cell geometry."""
default_extent(sizes::NTuple{N, Int}, global_periodic::NTuple{N, Bool}) where {N} =
    PaddedExtent{N}(sizes, ntuple(_ -> 0, N), sizes, global_periodic, :cell)

macro maybe_threads(flag, ex)
    return esc(:(($flag) ? (Base.Threads.@threads $ex) : $ex))
end

# size of the flux array staggered by one in dimension `d`, for N-dimensional data of shape `sizes`
@inline flux_size(sizes::NTuple, d, N) = ntuple(i -> sizes[i] + (i == d ? 1 : 0), min(N, 3))

"""
    WENO_step!(u::Tuple, args...; u_min::Tuple{Vararg{Real}}, u_max::Tuple{Vararg{Real}})

Advance multiple fields `u = (c1, c2, ...)` by one time step, all sharing the same
velocity and `WENOScheme` buffers. Each field is advected sequentially with its own
`u_min`/`u_max` bounds for the Zhang-Shu limiter.

This single method covers every dimensionality and backend (plain arrays,
KernelAbstractions, Chmy.jl): it just forwards each field and the remaining
positional arguments to the single-field `WENO_step!` method that matches at
runtime, so it needs no per-dimension or per-backend duplicate.
"""
function WENO_step!(u::Tuple, args...; u_min::Tuple{Vararg{Real}}, u_max::Tuple{Vararg{Real}})
    for i in eachindex(u)
        WENO_step!(u[i], args...; u_min = u_min[i], u_max = u_max[i])
    end
    return nothing
end
