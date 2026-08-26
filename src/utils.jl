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
    else
        return mod1(i - d, nx)       # Periodic
    end
end

@inline function right_index(i, d, nx, b::Integer)
    if b == 0
        return clamp(i + d, 1, nx)   # Dirichlet
    elseif b == 1
        return min(i + d, nx)        # Neumann
    else
        return mod1(i + d, nx)       # Periodic
    end
end

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
