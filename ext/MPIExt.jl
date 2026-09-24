module MPIExt

using FiniteDiffWENO5
using FiniteDiffWENO5: AbstractWENOTopology

import FiniteDiffWENO5: weno_ndims, weno_halo, weno_owned_size, weno_global_size,
    weno_global_offset, weno_periodic, weno_physical_low, weno_physical_high,
    weno_exchange_halo!, weno_allreduce_max, weno_allreduce_min, weno_cartesian_topology

using MPI

include("mpi/topology.jl")
include("mpi/halo.jl")

end # module MPIExt
