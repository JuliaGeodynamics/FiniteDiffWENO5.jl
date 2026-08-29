"""
    WENO_step!(u::T,
               v::NamedTuple{(:x, :y, :z), <:Tuple{Vararg{AbstractArray{<:Real}, 3}}},
               weno::WENOScheme,
               Δt, Δx, Δy, Δz;
               u_min = 0.0, u_max = 0.0) where T <: AbstractArray{<:Real, 3}

Advance the solution `u` by one time step using the 3rd-order SSP Runge-Kutta method with WENO5-Z as the spatial discretization in 3D.

# Arguments
- `u::T`: Current solution array to be updated in place.
- `v::NamedTuple{(:x, :y, :z), <:Tuple{Vararg{AbstractArray{<:Real}, 3}}}`: Velocity fields in each direction, possibly staggered depending on `weno.stag`.
- `weno::WENOScheme`: WENO scheme structure containing necessary parameters and temporary arrays.
- `Δt`: Time step size.
- `Δx`: Spatial grid size in the x-direction.
- `Δy`: Spatial grid size in the y-direction.
- `Δz`: Spatial grid size in the z-direction.
- `u_min`: Minimum value of `u` for the Zhang-Shu positivity limiter.
- `u_max`: Maximum value of `u` for the Zhang-Shu positivity limiter.

Citation: Borges et al. 2008: "An improved weighted essentially non-oscillatory scheme for hyperbolic conservation laws"
          doi:10.1016/j.jcp.2007.11.038
"""
function WENO_step!(u::T, v::NamedTuple{(:x, :y, :z), <:Tuple{Vararg{AbstractArray{<:Real}, 3}}}, weno::WENOScheme, Δt, Δx, Δy, Δz; u_min = 0.0, u_max = 0.0, lf_speeds = nothing) where {T <: AbstractArray{<:Real, 3}}

    nx, ny, nz = size(u, 1), size(u, 2), size(u, 3)
    Δx_, Δy_, Δz_ = inv(Δx), inv(Δy), inv(Δz)

    (; ut, du, stag, fl, fr, multithreading, upwind_mode, form) = weno

    if !upwind_mode
        # Velocity is prepared once, outside the three RK stages. The
        # Lax-Friedrichs speeds are likewise constant across all three stages.
        voperator = prepare_velocity!(weno, v)
        conservative = is_conservative(form)
        αx = conservative ? (lf_speeds === nothing ? lf_speed(voperator.x) : lf_speeds.x) : zero(eltype(u))
        αy = conservative ? (lf_speeds === nothing ? lf_speed(voperator.y) : lf_speeds.y) : zero(eltype(u))
        αz = conservative ? (lf_speeds === nothing ? lf_speed(voperator.z) : lf_speeds.z) : zero(eltype(u))
        scalar_operator_3D!(du, u, voperator, weno, nx, ny, nz, Δx_, Δy_, Δz_, u_min, u_max, αx, αy, αz)

        @inbounds @maybe_threads multithreading for I in CartesianIndices(ut)
            ut[I] = @muladd u[I] - Δt * du[I]
        end

        scalar_operator_3D!(du, ut, voperator, weno, nx, ny, nz, Δx_, Δy_, Δz_, u_min, u_max, αx, αy, αz)

        @inbounds @maybe_threads multithreading for I in CartesianIndices(ut)
            ut[I] = @muladd 0.75 * u[I] + 0.25 * ut[I] - 0.25 * Δt * du[I]
        end

        scalar_operator_3D!(du, ut, voperator, weno, nx, ny, nz, Δx_, Δy_, Δz_, u_min, u_max, αx, αy, αz)

        @inbounds @maybe_threads multithreading for I in CartesianIndices(u)
            u[I] = @muladd 1.0 / 3.0 * u[I] + 2.0 / 3.0 * ut[I] - (2.0 / 3.0) * Δt * du[I]
        end
    else
        # Use simple upwind scheme for debugging
        upwind_update_3D!(u, v, weno, nx, ny, nz, Δx_, Δy_, Δz_, Δt)
    end

    return nothing
end

# Multi-field advection (u = (c1, c2, ...) sharing v and WENOScheme buffers) is
# handled generically for every dimension and backend by the `WENO_step!(u::Tuple, ...)`
# method in src/utils.jl.
