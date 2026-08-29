
## Running on GPU

To use this package on a GPU, you need to use [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) to define the arrays and run the computations.
The KernelAbstractions.jl package introduces a macro-based programming model that simplifies vendor-specific GPU programming by abstracting away its complexities. This allows hardware-independent kernels to be written that can be compiled and executed on different device backends without altering the high-level code or compromising performance.


This will load an extension from FiniteDiffWENO5GPU.jl:

```julia
using KernelAbstractions
using FiniteDiffWENO5
```

That way, the two functions `WENOScheme()` and `WENO_step!()` will automatically dispatch to the array types if the argument `backend` is provided. See the KernelAbstractions.jl documentation for more details on how to set up the GPU backend.

Simplex-constrained material fractions use the same phase ordering and boundary
semantics on every backend:

```julia
scheme = MultiphaseWENOScheme(phases, backend;
    boundary = ntuple(_ -> PeriodicBC(), 4), stag = true)
WENO_step!(phases, velocity, scheme, Δt, Δx, Δy, backend)

# Chmy fields
scheme = MultiphaseWENOScheme(phases, grid;
    boundary = ntuple(_ -> PeriodicBC(), 4), stag = true)
WENO_step!(phases, velocity, scheme, Δt, Δx, Δy, grid, arch)
```

For Chmy schemes, tangential arrays carried by `PrescribedInflowBC` are copied to the
same backend as the phase fields during construction. It is therefore valid to supply
ordinary host arrays when constructing a GPU-backed Chmy scheme; the scheme retains the
backend copies used by its kernels.

KernelAbstractions CPU tests verify the backend-generic multiphase kernels. Actual GPU
execution requires a device-enabled downstream test job.

## Scalar transport on the GPU

Just like on the CPU, `WENOScheme` takes independent `form` and `stag` keywords, and
both backends dispatch on the same `form` tag:

```julia
weno = WENOScheme(u, backend; form = :conservative, boundary = (PeriodicBC(), PeriodicBC()), stag = true)
velocity = (; x = ones(backend, nx + 1))
WENO_step!(u, velocity, weno, Δt, Δx, backend)
```

See [Getting Started](@ref) for a full worked example (the same call pattern applies;
just construct arrays on the desired `backend` and pass it as an extra argument to
`WENOScheme` and `WENO_step!`).
