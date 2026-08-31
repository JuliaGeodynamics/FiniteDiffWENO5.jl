abstract type AbstractAdvectionForm end

"""Scalar conservative prescribed-linear transport, `∂u/∂t + ∇·(v u) = 0`."""
struct ConservativeForm <: AbstractAdvectionForm end

"""Scalar material transport, `∂u/∂t + v·∇u = 0`."""
struct NonConservativeForm <: AbstractAdvectionForm end

"""Whether a scalar scheme uses conservative split-flux transport."""
@inline is_conservative(::ConservativeForm) = true
@inline is_conservative(::NonConservativeForm) = false

"""Whether the first-order debug upwind operator supports a form/layout pair."""
@inline supports_upwind_mode(form::AbstractAdvectionForm, stag::Bool) =
    (is_conservative(form) && stag) || !is_conservative(form)

"""Validate opposite boundary pairs and return periodicity by velocity direction."""
function velocity_periodicity(boundary, names)
    return NamedTuple{names}(
        ntuple(
            d -> begin
                lo, hi = boundary[2d - 1], boundary[2d]
                periodic = lo isa PeriodicBC
                periodic == (hi isa PeriodicBC) || throw(
                    ArgumentError(
                        "staggered velocity direction $d requires paired periodic boundaries",
                    )
                )
                periodic
            end, length(names)
        )
    )
end

function advection_form(form::Symbol)
    form === :conservative && return ConservativeForm()
    form === :nonconservative && return NonConservativeForm()
    throw(ArgumentError("form must be :conservative or :nonconservative, got $form"))
end

"""
    validate_scalar_options(form, stag, lim_ZS, upwind_mode)

The conservative (Lax-Friedrichs split-flux) path has no bound-preserving flux
limiter yet, so `lim_ZS=true` combined with `form=:conservative` would silently
do nothing — the option looks accepted but the limiting never happens. Reject
that combination explicitly rather than letting a caller lose bound
preservation without any signal that it happened.
"""
function validate_scalar_options(form::AbstractAdvectionForm, stag::Bool, lim_ZS::Bool, upwind_mode::Bool)
    form isa ConservativeForm && lim_ZS && throw(
        ArgumentError(
            "lim_ZS=true has no effect for form=:conservative: the conservative " *
                "split-flux path has no bound-preserving limiter yet. Pass " *
                "form=:nonconservative for Zhang-Shu bound preservation.",
        )
    )
    upwind_mode && !supports_upwind_mode(form, stag) &&
        throw(
        ArgumentError(
            "upwind_mode=true supports form=:conservative only with stag=true; " *
                "form=:nonconservative supports either velocity layout",
        )
    )
    return nothing
end
