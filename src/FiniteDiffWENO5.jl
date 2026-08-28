module FiniteDiffWENO5

using MuladdMacro

export WENOScheme, WENO_step!
export MultiphaseWENOScheme
export AbstractAdvectionBoundary, PeriodicBC, ExtrapolateBC, PrescribedInflowBC, AdvectionBC

include("utils.jl")
include("boundaries.jl")
include("WENO5/cache.jl")
include("WENO5/reconstruction.jl")
include("limiter/zhang_shu_limiter.jl")
include("limiter/simplex_limiter.jl")
include("multiphase/reconstruction.jl")
include("multiphase/boundaries.jl")
include("multiphase/cache.jl")
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
