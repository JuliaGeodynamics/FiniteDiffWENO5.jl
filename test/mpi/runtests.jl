# Entry point for MPI tests, run under mpiexecjl. Never launched from
# `test/runtests.jl`, which must not start MPI itself.
#
# Usage: mpiexecjl -n P julia test/mpi/runtests.jl SUITE [SUITE...]

using Test
using MPI

const SUITES = Dict(
    "halo" => "test_halo.jl",
    "cfl" => "test_cfl_collective.jl",
    "scalar" => "test_scalar_advection.jl",
    "staggered" => "test_staggered_velocity.jl",
    "multiphase" => "test_multiphase.jl",
)

function main(args)
    known = join(sort(collect(keys(SUITES))), ", ")
    isempty(args) && error("usage: julia runtests.jl SUITE [SUITE...] (known: $known)")
    for a in args
        haskey(SUITES, a) || error("usage: unknown suite '$a' (known: $known)")
    end

    MPI.Init()
    try
        @testset verbose = true "FiniteDiffWENO5 MPI tests" begin
            for a in args
                @testset "$a" begin
                    include(joinpath(@__DIR__, SUITES[a]))
                end
            end
        end
    finally
        MPI.Finalize()
    end
    return nothing
end

main(ARGS)
