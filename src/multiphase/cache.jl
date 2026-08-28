@kwdef struct MultiphaseWENOScheme{T, NP, TArray, TFlux, TDiv, TBoundary} <: AbstractWENO
    # upwind and downwind constants
    γ::NTuple{3, T} = T.((0.1, 0.6, 0.3))
    # betas' constants
    χ::NTuple{2, T} = T.((13 / 12, 1 / 4))
    # stencil weights
    ζ::NTuple{5, T} = T.((1 / 3, 7 / 6, 11 / 6, 1 / 6, 5 / 6))
    # tolerance to machine precision of the type T
    ϵ::T = eps(T)
    # staggered grid or not (velocities on cell faces or cell centers)
    stag::Bool
    # boundary conditions
    boundary::TBoundary
    # multithreading
    multithreading::Bool
    # per-phase fluxes as NamedTuples of NTuple{NP} arrays
    fl::TFlux
    fr::TFlux
    # per-phase semi-discretisation of the advection term
    du::TArray
    # per-phase temporary array for the time stepping
    ut::TArray
    # cell-centred velocity divergence; `nothing` on the collocated path
    divv::TDiv
end

"""
    MultiphaseWENOScheme(phases::Tuple; boundary=nothing, stag=false, multithreading=true)

Structure containing the WENO5-Z constants and per-phase buffers for the *simultaneous*
advection of two or more material fractions constrained to the probability simplex,
`0 ≤ ϕₖ ≤ 1` and `Σₖϕₖ = 1`.

Unlike `WENOScheme` with a tuple of fields — which advects each field sequentially with
its own nonlinear weights and therefore does not preserve `Σₖϕₖ` — this scheme computes
one set of WENO-Z weights per face state from all phases together, reconstructs every
phase with those shared weights, and applies one common Zhang-Shu coefficient to the
whole face composition. Use `WENOScheme` for unrelated fields such as temperature or
tracers; use this type only for fractions of a whole.

# Arguments
- `phases::Tuple`: at least two 1D, 2D, or 3D cell-centred arrays with identical axes,
  element type, and concrete array type. Only used for type, size, and backend; values
  are not read.
- `boundary`: ordered tuple of `ExtrapolateBC()`, `PeriodicBC()`, or
  `PrescribedInflowBC(value)` conditions, or an `AdvectionBC`. Defaults to
  `ExtrapolateBC()` on every face. One boundary family applies to the whole phase vector
  on a given face.
- `stag::Bool`: whether velocities live on cell faces (`true`) or cell centers (`false`).
  Defaults to `false`.
- `multithreading::Bool`: whether to use multithreading (2D and 3D only). Defaults to `true`.

# Differences from `WENOScheme`
- No `lim_ZS` field. The simplex limiter is unconditional: the bound and sum invariants
  are the purpose of this type rather than an option.
- No `upwind_mode` field. The debugging upwind path is not supported.
- The step function takes no `u_min`/`u_max`. The bounds are fixed at `[0,1]` by the
  simplex definition.

# Fields
- `γ`, `χ`, `ζ`, `ϵ`: WENO5-Z constants, identical to `WENOScheme`.
- `stag::Bool`: staggered or collocated velocity layout.
- `boundary`: normalized tuple of typed advection boundary conditions.
- `multithreading::Bool`: whether to use multithreading.
- `fl::NamedTuple`, `fr::NamedTuple`: per-direction left/right face states, each an
  `NTuple{NP}` of arrays.
- `du::NTuple{NP}`: per-phase semi-discretisation of the advection term.
- `ut::NTuple{NP}`: per-phase temporary storage for the Runge-Kutta stages.
- `divv`: cell-centred velocity divergence when `stag=true`, `nothing` otherwise.
"""
function MultiphaseWENOScheme(
        phases::Tuple{Vararg{Any, NP}};
        boundary = nothing, stag::Bool = false, multithreading::Bool = true,
    ) where {NP}

    NP >= 2 || throw(ArgumentError(
            "MultiphaseWENOScheme requires at least two phases, got $NP. " *
                "Use WENOScheme for a single field."))

    all(p -> p isa AbstractArray, phases) || throw(ArgumentError(
            "MultiphaseWENOScheme requires a tuple of arrays, got " *
                "$(map(typeof, phases))"))

    c0 = first(phases)
    T = eltype(c0)
    N = ndims(c0)
    1 <= N <= 3 || throw(ArgumentError(
        "MultiphaseWENOScheme supports 1D, 2D, and 3D fields, got $(N)D"))

    for k in 2:NP
        p = phases[k]
        eltype(p) === T || throw(ArgumentError(
                "all phases must share an element type, phase 1 is $(T) but phase $k is " *
                    "$(eltype(p))"))
        ndims(p) == N || throw(DimensionMismatch(
                "all phases must share a dimensionality, phase 1 is $(N)D but phase $k is " *
                    "$(ndims(p))D"))
        axes(p) == axes(c0) || throw(DimensionMismatch(
                "all phases must share axes, phase 1 has $(axes(c0)) but phase $k has " *
                    "$(axes(p))"))
        typeof(p) === typeof(c0) || throw(ArgumentError(
                "all phases must share a concrete array type, phase 1 is $(typeof(c0)) " *
                    "but phase $k is $(typeof(p))"))
    end

    boundary === nothing && (boundary = ntuple(i -> ExtrapolateBC(), N * 2))
    boundary = validate_multiphase_boundary(boundary, N, size(c0), NP, T)

    # dimension labels
    labels = (:x, :y, :z)[1:min(N, 3)]
    sizes = size(c0)

    # allocate a zeroed buffer of the same array type as the phases
    zeros_like(dims) = fill!(similar(c0, T, dims), zero(T))

    # `Val(NP)` keeps the phase count a compile-time constant, so every buffer infers as a
    # concrete `NTuple` rather than an abstract `Tuple`.
    valNP = Val(NP)

    fl = NamedTuple{labels}(
        ntuple(min(N, 3)) do d
            ntuple(_ -> zeros_like(flux_size(sizes, d, N)), valNP)
        end
    )
    fr = NamedTuple{labels}(
        ntuple(min(N, 3)) do d
            ntuple(_ -> zeros_like(flux_size(sizes, d, N)), valNP)
        end
    )

    du = ntuple(_ -> zeros_like(sizes), valNP)
    ut = ntuple(_ -> zeros_like(sizes), valNP)

    # the collocated form telescopes to zero without a source, so no divergence is needed
    divv = stag ? zeros_like(sizes) : nothing

    return MultiphaseWENOScheme{T, NP, typeof(du), typeof(fl), typeof(divv), typeof(boundary)}(
        stag = stag, boundary = boundary, multithreading = multithreading,
        fl = fl, fr = fr, du = du, ut = ut, divv = divv,
    )
end

"""
    nphases(scheme::MultiphaseWENOScheme)

Number of phases carried by `scheme`, available as a compile-time constant.
"""
@inline nphases(::MultiphaseWENOScheme{T, NP}) where {T, NP} = NP
