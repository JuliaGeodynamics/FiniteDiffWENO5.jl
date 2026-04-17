"""
    WENO_step!(u::T,
               v::NamedTuple{(:x, :y, :z), <:Tuple{Vararg{Array{<:Real}, 3}}},
               weno::WENOScheme,
               Δt, Δx, Δy, Δz;
               u_min = 0.0, u_max = 0.0) where T <: AbstractArray{<:Real, 3}

Advance the solution `u` by one time step using the 3rd-order SSP Runge-Kutta method with WENO5-Z as the spatial discretization in 3D.

# Arguments
- `u::T`: Current solution array to be updated in place.
- `v::NamedTuple{(:x, :y, :z), <:Tuple{Vararg{Array{<:Real}, 3}}}`: Velocity fields in each direction, possibly staggered depending on `weno.stag`.
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
function WENO_step!(u::T, v::NamedTuple{(:x, :y, :z), <:Tuple{Vararg{Array{<:Real}, 3}}}, weno::WENOScheme, Δt, Δx, Δy, Δz; u_min = 0.0, u_max = 0.0) where {T <: Array{<:Real, 3}}

    nx, ny, nz = size(u, 1), size(u, 2), size(u, 3)
    Δx_, Δy_, Δz_ = inv(Δx), inv(Δy), inv(Δz)

    @unpack ut, du, stag, fl, fr, multithreading, upwind_mode = weno

    if !upwind_mode
        WENO_flux!(fl, fr, u, weno, nx, ny, nz, u_min, u_max)
        semi_discretisation_weno5!(du, v, weno, Δx_, Δy_, Δz_)

        @inbounds @maybe_threads multithreading for I in CartesianIndices(ut)
            ut[I] = @muladd u[I] - Δt * du[I]
        end

        WENO_flux!(fl, fr, ut, weno, nx, ny, nz, u_min, u_max)
        semi_discretisation_weno5!(du, v, weno, Δx_, Δy_, Δz_)

        @inbounds @maybe_threads multithreading for I in CartesianIndices(ut)
            ut[I] = @muladd 0.75 * u[I] + 0.25 * ut[I] - 0.25 * Δt * du[I]
        end

        WENO_flux!(fl, fr, ut, weno, nx, ny, nz, u_min, u_max)
        semi_discretisation_weno5!(du, v, weno, Δx_, Δy_, Δz_)

        @inbounds @maybe_threads multithreading for I in CartesianIndices(u)
            u[I] = @muladd 1.0 / 3.0 * u[I] + 2.0 / 3.0 * ut[I] - (2.0 / 3.0) * Δt * du[I]
        end
    else
        # Use simple upwind scheme for debugging
        upwind_update_3D!(u, v, weno, nx, ny, nz, Δx_, Δy_, Δz_, Δt)
    end

    return nothing
end

"""
    WENO_step!(u::Tuple{Vararg{Array{<:Real, 3}}},
               v::NamedTuple{(:x, :y, :z), <:Tuple{Vararg{Array{<:Real}, 3}}},
               weno::WENOScheme,
               Δt, Δx, Δy, Δz;
               u_min::Tuple{Vararg{Real}},
               u_max::Tuple{Vararg{Real}})

Advance multiple fields `u = (c1, c2, ...)` by one time step, all sharing the same velocity `v` and `WENOScheme` buffers.
Each field is advected sequentially with its own `u_min` / `u_max` bounds for the Zhang-Shu limiter.
"""
function WENO_step!(u::Tuple{Vararg{Array{<:Real, 3}}}, v::NamedTuple{(:x, :y, :z), <:Tuple{Vararg{Array{<:Real}, 3}}}, weno::WENOScheme, Δt, Δx, Δy, Δz; u_min::Tuple{Vararg{Real}}, u_max::Tuple{Vararg{Real}})
    for i in eachindex(u)
        WENO_step!(u[i], v, weno, Δt, Δx, Δy, Δz; u_min = u_min[i], u_max = u_max[i])
    end
    return nothing
end
