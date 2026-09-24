module FiniteDiffWENO5

using MuladdMacro

export WENOScheme, WENO_step!
export MultiphaseWENOScheme
export AbstractAdvectionBoundary, PeriodicBC, ExtrapolateBC, PrescribedInflowBC, AdvectionBC
export weno_cartesian_topology, allocate_weno_field, weno_global_ranges
export weno_cfl_dt, weno_substeps

include("utils.jl")
include("boundaries.jl")
include("topology_interface.jl")
include("advection_form.jl")

"""
    weno_cartesian_topology(global_dims; comm, halo, dims, periodic)

Build the real MPI Cartesian topology. This is a stub in core. Its methods
live in the `MPI` weak-dependency extension (`ext/MPIExt.jl`); calling it
before `using MPI` throws naming the missing extension, the same pattern
every [`AbstractWENOTopology`](@ref) accessor uses.
"""
function weno_cartesian_topology(global_dims; kwargs...)
    throw(
        ArgumentError(
            "weno_cartesian_topology has no method for these arguments. Load MPI " *
                "(`using MPI`) to activate FiniteDiffWENO5's MPI extension, which " *
                "provides the real implementation."
        )
    )
end

include("WENO5/cache.jl")
include("WENO5/reconstruction.jl")
include("WENO5/conservative_flux.jl")
include("limiter/zhang_shu_limiter.jl")
include("limiter/simplex_limiter.jl")
include("multiphase/reconstruction.jl")
include("multiphase/boundaries.jl")
include("multiphase/cache.jl")
include("staggered_velocity_interpolation.jl")
include("distributed_step.jl")
include("multi_field_time_stepping.jl")
include("1D/semi_discretisation_1D.jl")
include("1D/time_stepping.jl")
include("1D/multiphase_semi_discretisation_1D.jl")
include("1D/multiphase_time_stepping.jl")
include("2D/semi_discretisation_2D.jl")
include("2D/time_stepping.jl")
include("2D/multiphase_semi_discretisation_2D.jl")
include("2D/multiphase_time_stepping.jl")
include("3D/semi_discretisation_3D.jl")
include("3D/time_stepping.jl")
include("3D/multiphase_semi_discretisation_3D.jl")
include("3D/multiphase_time_stepping.jl")


end # module FiniteDiffWENO5
