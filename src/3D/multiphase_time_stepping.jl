"""
    WENO_step!(phases, velocity, scheme::MultiphaseWENOScheme, Δt, Δx, Δy, Δz)

Advance a three-dimensional material composition with simultaneous WENO5-Z
reconstruction and SSP-RK3. `phases` is updated in place and must initially satisfy
the probability-simplex constraints. No `u_min` or `u_max` keywords are accepted.
"""
function WENO_step!(
        phases::Tuple{A, Vararg{A, M}},
        v::NamedTuple{(:x, :y, :z), <:Tuple{Vararg{AbstractArray{<:Real}, 3}}},
        scheme::MultiphaseWENOScheme{T, NP}, Δt, Δx, Δy, Δz,
    ) where {M, A <: AbstractArray{<:Real, 3}, T, NP}
    M + 1 == NP || throw(
        DimensionMismatch(
            "scheme was built for $NP phases but $(M + 1) were given"
        )
    )

    nx, ny, nz = size(phases[1])
    Δx_, Δy_, Δz_ = inv(Δx), inv(Δy), inv(Δz)
    (; ut, du, multithreading) = scheme

    multiphase_WENO_flux!(phases, scheme, nx, ny, nz)
    multiphase_semi_discretisation!(du, phases, v, scheme, Δx_, Δy_, Δz_)
    @inbounds @maybe_threads multithreading for I in CartesianIndices(ut[1])
        for q in 1:NP
            ut[q][I] = @muladd phases[q][I] - Δt * du[q][I]
        end
    end

    multiphase_WENO_flux!(ut, scheme, nx, ny, nz)
    multiphase_semi_discretisation!(du, ut, v, scheme, Δx_, Δy_, Δz_)
    @inbounds @maybe_threads multithreading for I in CartesianIndices(ut[1])
        for q in 1:NP
            ut[q][I] = @muladd 0.75 * phases[q][I] + 0.25 * ut[q][I] -
                0.25 * Δt * du[q][I]
        end
    end

    multiphase_WENO_flux!(ut, scheme, nx, ny, nz)
    multiphase_semi_discretisation!(du, ut, v, scheme, Δx_, Δy_, Δz_)
    @inbounds @maybe_threads multithreading for I in CartesianIndices(phases[1])
        for q in 1:NP
            phases[q][I] = @muladd 1.0 / 3.0 * phases[q][I] +
                2.0 / 3.0 * ut[q][I] - (2.0 / 3.0) * Δt * du[q][I]
        end
    end
    return nothing
end
