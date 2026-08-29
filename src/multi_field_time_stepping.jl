"""
CPU-only tuple overload that prepares one staggered material velocity for all
fields, instead of re-running `prepare_velocity!` once per field.

Constrained to `A <: Array` rather than a bare `u::Tuple`: `WENOScheme` is the same
struct for every backend (CPU, KernelAbstractions, Chmy), so an unconstrained method
here would out-specificity the generic per-field forwarder in `src/utils.jl` for
every backend, not just this one — silently reaching `prepare_velocity!`'s
CPU-only scalar-indexing loop (`eno5_face_to_center!`'s `for i in eachindex(...)`)
for GPU-backed schemes, which throws under GPUArrays.jl's `allowscalar(false)` (or
runs pathologically slowly if allowed). Restricting to plain `Array` excludes every
device array type (`CuArray`, `ROCArray`, ...) and Chmy's `Field` wrapper alike,
letting those fall through to the generic forwarder, which dispatches each field to
its own backend-correct single-field `WENO_step!` (and that backend's own device
`prepare_velocity_*!` kernel) instead.
"""
function WENO_step!(u::Tuple{A, Vararg{A, M}}, velocity, weno::WENOScheme, args...;
                    u_min::Tuple{Vararg{Real}}, u_max::Tuple{Vararg{Real}}) where {A <: Array, M}
    velocity_step = prepare_velocity!(weno, velocity)
    # KA's CPU backend also stores fields in `Array`s, but its scalar methods take
    # an additional backend positional argument and do not accept this private
    # CPU-only keyword. A plain CPU step has Δt plus one spacing per dimension.
    cpu_step = length(args) == ndims(first(u)) + 1
    speeds = cpu_step ? lf_speeds(weno.form, velocity_step) : nothing
    for i in eachindex(u)
        if cpu_step
            WENO_step!(
                u[i], velocity_step, weno, args...;
                u_min = u_min[i], u_max = u_max[i], lf_speeds = speeds,
            )
        else
            WENO_step!(u[i], velocity_step, weno, args...; u_min = u_min[i], u_max = u_max[i])
        end
    end
    return nothing
end
