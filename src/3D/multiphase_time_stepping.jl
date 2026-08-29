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
    voperator = prepare_velocity!(scheme, v)
    valNP = Val(NP)

    multiphase_WENO_flux!(phases, scheme, nx, ny, nz)
    multiphase_material_semi_discretisation!(du, voperator, scheme, Δx_, Δy_, Δz_)
    @inbounds @maybe_threads multithreading for I in CartesianIndices(ut[1])
        initial = ntuple(q -> phases[q][I], valNP)
        duT = ntuple(q -> du[q][I], valNP)
        updated = simplex_rk_stage(initial, initial, duT, zero(T), one(T), Δt)
        for q in 1:NP
            ut[q][I] = updated[q]
        end
    end

    multiphase_WENO_flux!(ut, scheme, nx, ny, nz)
    multiphase_material_semi_discretisation!(du, voperator, scheme, Δx_, Δy_, Δz_)
    @inbounds @maybe_threads multithreading for I in CartesianIndices(ut[1])
        initial = ntuple(q -> phases[q][I], valNP)
        stage = ntuple(q -> ut[q][I], valNP)
        duT = ntuple(q -> du[q][I], valNP)
        updated = simplex_rk_stage(initial, stage, duT, T(0.75), T(0.25), Δt)
        for q in 1:NP
            ut[q][I] = updated[q]
        end
    end

    multiphase_WENO_flux!(ut, scheme, nx, ny, nz)
    multiphase_material_semi_discretisation!(du, voperator, scheme, Δx_, Δy_, Δz_)
    @inbounds @maybe_threads multithreading for I in CartesianIndices(phases[1])
        initial = ntuple(q -> phases[q][I], valNP)
        stage = ntuple(q -> ut[q][I], valNP)
        duT = ntuple(q -> du[q][I], valNP)
        updated = simplex_rk_stage(initial, stage, duT, T(1 / 3), T(2 / 3), Δt)
        for q in 1:NP
            phases[q][I] = updated[q]
        end
    end
    return nothing
end
