module FiniteDiffWENO5

using MuladdMacro

export WENOScheme, WENO_step!
export AbstractAdvectionBoundary, PeriodicBC, ExtrapolateBC, PrescribedInflowBC, AdvectionBC

include("utils.jl")
include("boundaries.jl")
include("WENO5/cache.jl")
include("WENO5/reconstruction.jl")
include("limiter/zhang_shu_limiter.jl")
include("1D/semi_discretisation_1D.jl")
include("1D/time_stepping.jl")
include("2D/semi_discretisation_2D.jl")
include("2D/time_stepping.jl")
include("3D/semi_discretisation_3D.jl")
include("3D/time_stepping.jl")


end # module FiniteDiffWENO5
