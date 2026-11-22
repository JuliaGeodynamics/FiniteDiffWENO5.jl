"""
CPU tuple step: prepares the velocity once for all fields. Restricted to `Array`
because the CPU `prepare_velocity!` must not see device arrays or Chmy fields.
"""
function WENO_step!(
        u::Tuple{A, Vararg{A, M}}, velocity, weno::WENOScheme, args...;
        u_min::Tuple{Vararg{Real}}, u_max::Tuple{Vararg{Real}}
    ) where {A <: Array, M}
    # KA's CPU backend also uses `Array`s but takes an extra backend argument.
    cpu_step = length(args) == ndims(first(u)) + 1
    cpu_step || return step_fields_shared!(u, velocity, weno, args, u_min, u_max)
    velocity_step = prepare_velocity!(weno, velocity)
    for i in eachindex(u)
        WENO_step!(
            u[i], velocity_step, weno, args...;
            u_min = u_min[i], u_max = u_max[i], face_velocity = velocity,
        )
    end
    return nothing
end

"""Tuple step for the other backends, which prepare the velocity themselves."""
function WENO_step!(
        u::Tuple, velocity, weno::WENOScheme, args...;
        u_min::Tuple{Vararg{Real}}, u_max::Tuple{Vararg{Real}}
    )
    return step_fields_shared!(u, velocity, weno, args, u_min, u_max)
end

# The first field prepares `weno.vcenter`; the others reuse it, with the raw
# velocity as face velocity. Upwind mode needs the raw velocity, so never shares.
function step_fields_shared!(u, velocity, weno, args, u_min, u_max)
    WENO_step!(u[1], velocity, weno, args...; u_min = u_min[1], u_max = u_max[1])
    share = weno.stag && !weno.upwind_mode && weno.vcenter !== nothing && velocity !== weno.vcenter
    v = share ? weno.vcenter : velocity
    face_velocity = share ? velocity : nothing
    for i in 2:length(u)
        WENO_step!(u[i], v, weno, args...; u_min = u_min[i], u_max = u_max[i], face_velocity)
    end
    return nothing
end
