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
    M + 1 == NP || throw(
        DimensionMismatch(
            "scheme was built for $NP phases but $(M + 1) were given"
        )
    )

    nx, ny = size(phases[1])
    Δx_, Δy_ = inv(Δx), inv(Δy)
    (; ut, du, multithreading) = scheme
    voperator = prepare_velocity!(scheme, v)
    valNP = Val(NP)

    multiphase_WENO_flux!(phases, scheme, nx, ny)
    multiphase_material_semi_discretisation!(du, voperator, scheme, Δx_, Δy_)
    @inbounds @maybe_threads multithreading for I in CartesianIndices(ut[1])
        initial = ntuple(k -> phases[k][I], valNP)
        duT = ntuple(k -> du[k][I], valNP)
        updated = simplex_rk_stage(initial, initial, duT, zero(T), one(T), Δt)
        for k in 1:NP
            ut[k][I] = updated[k]
        end
    end

    multiphase_WENO_flux!(ut, scheme, nx, ny)
    multiphase_material_semi_discretisation!(du, voperator, scheme, Δx_, Δy_)
    @inbounds @maybe_threads multithreading for I in CartesianIndices(ut[1])
        initial = ntuple(k -> phases[k][I], valNP)
        stage = ntuple(k -> ut[k][I], valNP)
        duT = ntuple(k -> du[k][I], valNP)
        updated = simplex_rk_stage(initial, stage, duT, T(0.75), T(0.25), Δt)
        for k in 1:NP
            ut[k][I] = updated[k]
        end
    end

    multiphase_WENO_flux!(ut, scheme, nx, ny)
    multiphase_material_semi_discretisation!(du, voperator, scheme, Δx_, Δy_)
    @inbounds @maybe_threads multithreading for I in CartesianIndices(phases[1])
        initial = ntuple(k -> phases[k][I], valNP)
        stage = ntuple(k -> ut[k][I], valNP)
        duT = ntuple(k -> du[k][I], valNP)
        updated = simplex_rk_stage(initial, stage, duT, T(1 / 3), T(2 / 3), Δt)
        for k in 1:NP
            phases[k][I] = updated[k]
        end
    end
    return nothing
end
