using Test
using FiniteDiffWENO5
using KernelAbstractions
using Chmy

function runtests()
    files = readdir(@__DIR__)
    test_files = filter(startswith("test_"), files)

    for f in test_files
        if !isdir(f)
            include(f)
        end
    end
    return
end

@testset verbose = true "All tests" begin
    runtests()
end
