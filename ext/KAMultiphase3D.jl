@kernel inbounds = true function multiphase_WENO_flux_KA_3D_x!(
        fl, fr, state, boundary, nx, χ, γ, ζ, ϵ, phase_count::Val{NP}, g, O
    ) where {NP}
    I = @index(Global, NTuple)
    i, j, k = I + O
    iwww = left_index(i, 3, nx, boundary[1]); iww = left_index(i, 2, nx, boundary[1])
    iw = left_index(i, 1, nx, boundary[1]); ie = right_index(i, 0, nx, boundary[2])
    iee = right_index(i, 1, nx, boundary[2]); ieee = right_index(i, 2, nx, boundary[2])
    sl = ntuple(
        q -> (
            state[q][iwww, j, k], state[q][iww, j, k],
            state[q][iw, j, k], state[q][ie, j, k], state[q][iee, j, k],
        ), phase_count
    )
    sr = ntuple(
        q -> (
            state[q][iww, j, k], state[q][iw, j, k],
            state[q][ie, j, k], state[q][iee, j, k], state[q][ieee, j, k],
        ), phase_count
    )
    up = limit_simplex(
        multiphase_reconstruction_upwind(sl, χ, γ, ζ, ϵ),
        ntuple(q -> state[q][iw, j, k], phase_count)
    )
    dn = limit_simplex(
        multiphase_reconstruction_downwind(sr, χ, γ, ζ, ϵ),
        ntuple(q -> state[q][ie, j, k], phase_count)
    )
    for q in 1:NP
        fl[q][i, j, k] = up[q]
        fr[q][i, j, k] = dn[q]
    end
    if i == 1 && 1 <= j <= size(state[1], 2) && 1 <= k <= size(state[1], 3) &&
            boundary[1] isa PrescribedInflowBC
        for q in 1:NP
            fl[q][i, j, k] = multiphase_inflow_value(boundary[1], q, j, k)
        end
    end
    if i == nx + 1 && 1 <= j <= size(state[1], 2) && 1 <= k <= size(state[1], 3) &&
            boundary[2] isa PrescribedInflowBC
        for q in 1:NP
            fr[q][i, j, k] = multiphase_inflow_value(boundary[2], q, j, k)
        end
    end
end

@kernel inbounds = true function multiphase_WENO_flux_KA_3D_y!(
        fl, fr, state, boundary, ny, χ, γ, ζ, ϵ, phase_count::Val{NP}, g, O
    ) where {NP}
    I = @index(Global, NTuple)
    i, j, k = I + O
    jwww = left_index(j, 3, ny, boundary[3]); jww = left_index(j, 2, ny, boundary[3])
    jw = left_index(j, 1, ny, boundary[3]); je = right_index(j, 0, ny, boundary[4])
    jee = right_index(j, 1, ny, boundary[4]); jeee = right_index(j, 2, ny, boundary[4])
    sl = ntuple(
        q -> (
            state[q][i, jwww, k], state[q][i, jww, k],
            state[q][i, jw, k], state[q][i, je, k], state[q][i, jee, k],
        ), phase_count
    )
    sr = ntuple(
        q -> (
            state[q][i, jww, k], state[q][i, jw, k],
            state[q][i, je, k], state[q][i, jee, k], state[q][i, jeee, k],
        ), phase_count
    )
    up = limit_simplex(
        multiphase_reconstruction_upwind(sl, χ, γ, ζ, ϵ),
        ntuple(q -> state[q][i, jw, k], phase_count)
    )
    dn = limit_simplex(
        multiphase_reconstruction_downwind(sr, χ, γ, ζ, ϵ),
        ntuple(q -> state[q][i, je, k], phase_count)
    )
    for q in 1:NP
        fl[q][i, j, k] = up[q]
        fr[q][i, j, k] = dn[q]
    end
    if j == 1 && 1 <= i <= size(state[1], 1) && 1 <= k <= size(state[1], 3) &&
            boundary[3] isa PrescribedInflowBC
        for q in 1:NP
            fl[q][i, j, k] = multiphase_inflow_value(boundary[3], q, i, k)
        end
    end
    if j == ny + 1 && 1 <= i <= size(state[1], 1) && 1 <= k <= size(state[1], 3) &&
            boundary[4] isa PrescribedInflowBC
        for q in 1:NP
            fr[q][i, j, k] = multiphase_inflow_value(boundary[4], q, i, k)
        end
    end
end

@kernel inbounds = true function multiphase_WENO_flux_KA_3D_z!(
        fl, fr, state, boundary, nz, χ, γ, ζ, ϵ, phase_count::Val{NP}, g, O
    ) where {NP}
    I = @index(Global, NTuple)
    i, j, k = I + O
    kwww = left_index(k, 3, nz, boundary[5]); kww = left_index(k, 2, nz, boundary[5])
    kw = left_index(k, 1, nz, boundary[5]); ke = right_index(k, 0, nz, boundary[6])
    kee = right_index(k, 1, nz, boundary[6]); keee = right_index(k, 2, nz, boundary[6])
    sl = ntuple(
        q -> (
            state[q][i, j, kwww], state[q][i, j, kww],
            state[q][i, j, kw], state[q][i, j, ke], state[q][i, j, kee],
        ), phase_count
    )
    sr = ntuple(
        q -> (
            state[q][i, j, kww], state[q][i, j, kw],
            state[q][i, j, ke], state[q][i, j, kee], state[q][i, j, keee],
        ), phase_count
    )
    up = limit_simplex(
        multiphase_reconstruction_upwind(sl, χ, γ, ζ, ϵ),
        ntuple(q -> state[q][i, j, kw], phase_count)
    )
    dn = limit_simplex(
        multiphase_reconstruction_downwind(sr, χ, γ, ζ, ϵ),
        ntuple(q -> state[q][i, j, ke], phase_count)
    )
    for q in 1:NP
        fl[q][i, j, k] = up[q]
        fr[q][i, j, k] = dn[q]
    end
    if k == 1 && 1 <= i <= size(state[1], 1) && 1 <= j <= size(state[1], 2) &&
            boundary[5] isa PrescribedInflowBC
        for q in 1:NP
            fl[q][i, j, k] = multiphase_inflow_value(boundary[5], q, i, j)
        end
    end
    if k == nz + 1 && 1 <= i <= size(state[1], 1) && 1 <= j <= size(state[1], 2) &&
            boundary[6] isa PrescribedInflowBC
        for q in 1:NP
            fr[q][i, j, k] = multiphase_inflow_value(boundary[6], q, i, j)
        end
    end
end

@kernel inbounds = true function multiphase_semi_collocated_KA_3D!(
        du, fl, fr, v, Δx_, Δy_, Δz_, ::Val{NP}, g, O
    ) where {NP}
    I = @index(Global, Cartesian)
    I = I + O
    i, j, k = I[1], I[2], I[3]
    for q in 1:NP
        du[q][I] = @muladd max(v.x[I], 0) *
            (fl.x[q][i + 1, j, k] - fl.x[q][I]) * Δx_ +
            min(v.x[I], 0) * (fr.x[q][i + 1, j, k] - fr.x[q][I]) * Δx_ +
            max(v.y[I], 0) * (fl.y[q][i, j + 1, k] - fl.y[q][I]) * Δy_ +
            min(v.y[I], 0) * (fr.y[q][i, j + 1, k] - fr.y[q][I]) * Δy_ +
            max(v.z[I], 0) * (fl.z[q][i, j, k + 1] - fl.z[q][I]) * Δz_ +
            min(v.z[I], 0) * (fr.z[q][i, j, k + 1] - fr.z[q][I]) * Δz_
    end
end

"""Material transport in 3D; see the 2D counterpart for the design rationale."""
function launch_multiphase_stage_KA_3D!(
        dest, initial, stage, du, fl, fr, vcell, boundary,
        nx, ny, nz, χ, γ, ζ, ϵ, Δx_, Δy_, Δz_, a, b, Δt, phase_count, backend,
        fx, fy, fz, semi, limited_update,
    )
    fx(
        fl.x, fr.x, stage, boundary, nx, χ, γ, ζ, ϵ, phase_count, nothing, Offset0;
        ndrange = size(fl.x[1])
    )
    fy(
        fl.y, fr.y, stage, boundary, ny, χ, γ, ζ, ϵ, phase_count, nothing, Offset0;
        ndrange = size(fl.y[1])
    )
    fz(
        fl.z, fr.z, stage, boundary, nz, χ, γ, ζ, ϵ, phase_count, nothing, Offset0;
        ndrange = size(fl.z[1])
    )
    synchronize(backend)
    semi(du, fl, fr, vcell, Δx_, Δy_, Δz_, phase_count, nothing, Offset0; ndrange = size(du[1]))
    synchronize(backend)
    limited_update(
        dest, initial, stage, du, a, b, Δt, phase_count, nothing, Offset0;
        ndrange = size(dest[1])
    )
    synchronize(backend)
    return nothing
end

"""Interpolate a staggered multiphase velocity to cell centres once per step (3D)."""
function prepare_multiphase_velocity_KA_3D!(scheme, v, nx, ny, nz, backend)
    scheme.stag || return v
    scheme.vcenter === nothing && return v
    FiniteDiffWENO5.validate_staggered_velocity!(scheme.vcenter, v; periodic = scheme.vperiodic)
    px, py, pz = scheme.vperiodic.x, scheme.vperiodic.y, scheme.vperiodic.z
    cx, cy, cz = scheme.vcenter.x, scheme.vcenter.y, scheme.vcenter.z
    kx = nx >= FiniteDiffWENO5.eno5_minimum_cells(px) ?
        eno5_face_to_center_KA_3D_x!(backend) : linear_face_to_center_KA_3D_x!(backend)
    ky = ny >= FiniteDiffWENO5.eno5_minimum_cells(py) ?
        eno5_face_to_center_KA_3D_y!(backend) : linear_face_to_center_KA_3D_y!(backend)
    kz = nz >= FiniteDiffWENO5.eno5_minimum_cells(pz) ?
        eno5_face_to_center_KA_3D_z!(backend) : linear_face_to_center_KA_3D_z!(backend)
    kx(cx, v.x, nx, px, nothing, Offset0, ndrange = size(cx))
    ky(cy, v.y, ny, py, nothing, Offset0, ndrange = size(cy))
    kz(cz, v.z, nz, pz, nothing, Offset0, ndrange = size(cz))
    synchronize(backend)
    return (; x = cx, y = cy, z = cz)
end

if nameof(@__MODULE__) == :KAExt
    function WENO_step!(
            phases::Tuple{A, Vararg{A, M}},
            v::NamedTuple{(:x, :y, :z), <:Tuple{Vararg{AbstractArray{<:Real}, 3}}},
            scheme::MultiphaseWENOScheme{T, NP}, Δt, Δx, Δy, Δz, backend::Backend,
        ) where {M, A <: AbstractArray{<:Real, 3}, T, NP}
        M + 1 == NP || throw(
            DimensionMismatch(
                "scheme was built for $NP phases but $(M + 1) were given"
            )
        )
        for q in 1:NP
            @assert get_backend(phases[q]) == backend
        end
        @assert get_backend(v.x) == backend
        @assert get_backend(v.y) == backend
        @assert get_backend(v.z) == backend

        (; fl, fr, ut, du, boundary, χ, γ, ζ, ϵ) = scheme
        nx, ny, nz = size(phases[1])
        Δx_, Δy_, Δz_ = inv(Δx), inv(Δy), inv(Δz)
        phase_count = Val(NP)
        vcell = prepare_multiphase_velocity_KA_3D!(scheme, v, nx, ny, nz, backend)
        fx = multiphase_WENO_flux_KA_3D_x!(backend)
        fy = multiphase_WENO_flux_KA_3D_y!(backend)
        fz = multiphase_WENO_flux_KA_3D_z!(backend)
        semi = multiphase_semi_collocated_KA_3D!(backend)
        limited_update = multiphase_RK_limited_update_KA!(backend)
        launch_multiphase_stage_KA_3D!(
            ut, phases, phases, du, fl, fr, vcell,
            boundary, nx, ny, nz, χ, γ, ζ, ϵ, Δx_, Δy_, Δz_, 0.0, 1.0, Δt,
            phase_count, backend, fx, fy, fz, semi, limited_update
        )
        launch_multiphase_stage_KA_3D!(
            ut, phases, ut, du, fl, fr, vcell,
            boundary, nx, ny, nz, χ, γ, ζ, ϵ, Δx_, Δy_, Δz_, 0.75, 0.25, Δt,
            phase_count, backend, fx, fy, fz, semi, limited_update
        )
        launch_multiphase_stage_KA_3D!(
            phases, phases, ut, du, fl, fr, vcell,
            boundary, nx, ny, nz, χ, γ, ζ, ϵ, Δx_, Δy_, Δz_, 1.0 / 3.0,
            2.0 / 3.0, Δt, phase_count, backend, fx, fy, fz, semi, limited_update
        )
        return nothing
    end
end
