"""
    WENO_step!(phases, velocity, scheme::MultiphaseWENOScheme, Δt, Δx, Δy)

Advance a two-dimensional material composition with simultaneous WENO5-Z
reconstruction and SSP-RK3. `phases` is updated in place and must initially satisfy
the probability-simplex constraints. No `u_min` or `u_max` keywords are accepted.
"""
function WENO_step!(
        phases::Tuple{A, Vararg{A, M}},
        v::NamedTuple{(:x, :y), <:Tuple{Vararg{AbstractArray{<:Real}, 2}}},
        scheme::MultiphaseWENOScheme{T, NP}, Δt, Δx, Δy,
    ) where {M, A <: AbstractMatrix{<:Real}, T, NP}
    M + 1 == NP || throw(DimensionMismatch(
        "scheme was built for $NP phases but $(M + 1) were given"))

    nx, ny = size(phases[1])
    Δx_, Δy_ = inv(Δx), inv(Δy)
    (; ut, du, multithreading) = scheme

    multiphase_WENO_flux!(phases, scheme, nx, ny)
    multiphase_semi_discretisation!(du, phases, v, scheme, Δx_, Δy_)
    @inbounds @maybe_threads multithreading for I in CartesianIndices(ut[1])
        for k in 1:NP
            ut[k][I] = @muladd phases[k][I] - Δt * du[k][I]
        end
    end

    multiphase_WENO_flux!(ut, scheme, nx, ny)
    multiphase_semi_discretisation!(du, ut, v, scheme, Δx_, Δy_)
    @inbounds @maybe_threads multithreading for I in CartesianIndices(ut[1])
        for k in 1:NP
            ut[k][I] = @muladd 0.75 * phases[k][I] + 0.25 * ut[k][I] -
                0.25 * Δt * du[k][I]
        end
    end

    multiphase_WENO_flux!(ut, scheme, nx, ny)
    multiphase_semi_discretisation!(du, ut, v, scheme, Δx_, Δy_)
    @inbounds @maybe_threads multithreading for I in CartesianIndices(phases[1])
        for k in 1:NP
            phases[k][I] = @muladd 1.0 / 3.0 * phases[k][I] +
                2.0 / 3.0 * ut[k][I] - (2.0 / 3.0) * Δt * du[k][I]
        end
    end
    return nothing
end
