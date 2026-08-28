"""
    WENO_step!(phases::Tuple, v::NamedTuple{(:x,)}, scheme::MultiphaseWENOScheme, Δt, Δx)

Advance a phase vector by one time step with 3rd-order SSP Runge-Kutta and simultaneous
WENO5-Z reconstruction in 1D.

Every phase is reconstructed from the same stage state before any phase advances, so all
three stages see a consistent composition. The result stays inside the probability
simplex — `0 ≤ ϕₖ ≤ 1` and `Σₖϕₖ = 1` — under the same explicit CFL assumptions the
scalar operator requires.

# Arguments
- `phases::Tuple`: the material fractions, updated in place. Must match the phase count
  the `scheme` was built with.
- `v`: velocity, at cell faces when `scheme.stag` is `true` and cell centers otherwise.
- `scheme::MultiphaseWENOScheme`: constants and per-phase buffers.
- `Δt`, `Δx`: time step and grid spacing.

Unlike the scalar `WENO_step!`, this takes no `u_min`/`u_max`: the bounds are fixed at
`[0,1]` by the simplex definition rather than supplied per field.

See also [`MultiphaseWENOScheme`](@ref).
"""
function WENO_step!(
        phases::Tuple{A, Vararg{A, M}},
        v::NamedTuple{(:x,), <:Tuple{<:AbstractVector{<:Real}}},
        scheme::MultiphaseWENOScheme{T, NP}, Δt, Δx,
    ) where {M, A <: AbstractVector{<:Real}, T, NP}

    M + 1 == NP || throw(
        DimensionMismatch(
            "scheme was built for $NP phases but $(M + 1) were given"
        )
    )

    nx = size(phases[1], 1)
    Δx_ = inv(Δx)

    (; ut, du, multithreading) = scheme

    # stage 1
    multiphase_WENO_flux!(phases, scheme, nx)
    multiphase_semi_discretisation!(du, phases, v, scheme, Δx_)
    @inbounds @maybe_threads multithreading for i in axes(ut[1], 1)
        for k in 1:NP
            ut[k][i] = @muladd phases[k][i] - Δt * du[k][i]
        end
    end

    # stage 2
    multiphase_WENO_flux!(ut, scheme, nx)
    multiphase_semi_discretisation!(du, ut, v, scheme, Δx_)
    @inbounds @maybe_threads multithreading for i in axes(ut[1], 1)
        for k in 1:NP
            ut[k][i] = @muladd 0.75 * phases[k][i] + 0.25 * ut[k][i] - 0.25 * Δt * du[k][i]
        end
    end

    # stage 3
    multiphase_WENO_flux!(ut, scheme, nx)
    multiphase_semi_discretisation!(du, ut, v, scheme, Δx_)
    @inbounds @maybe_threads multithreading for i in axes(phases[1], 1)
        for k in 1:NP
            phases[k][i] = @muladd 1.0 / 3.0 * phases[k][i] + 2.0 / 3.0 * ut[k][i] -
                (2.0 / 3.0) * Δt * du[k][i]
        end
    end

    return nothing
end
