# [Background](@id background)

The weighted essentially non-oscillatory (WENO) schemes form a class of high-order accurate numerical methods for solving hyperbolic partial differential equations (PDEs). They are particularly effective at resolving sharp gradients and discontinuities without introducing spurious oscillations. The WENO schemes achieve this by constructing nonlinear convex combinations of lower-order candidate polynomials, where the weights are determined by the local smoothness of the solution. This approach allows the method to retain high-order accuracy in smooth regions while automatically reducing to lower-order, more stable behavior near discontinuities. The WENO framework was first introduced by [Liu et al. 1994](https://doi.org/10.1006/jcph.1994.1187]), building upon the earlier essentially non-oscillatory (ENO) schemes developed by [Harten et al. 1987](https://doi.org/10.1016/0021-9991(87)90031-3).

In practice, WENO schemes can be formulated either in a finite-volume or finite-difference framework. In this package, we adopt the finite-difference formulation, which is particularly well-suited for problems defined on structured grids due to its simplicity and computational efficiency. The time integration is based on a third order strong stability preserving Runge-Kutta (SSP-RK3) method.

The implementation of a finite-difference WENO scheme involves the following main steps:

2. **Smoothness Indicators**: For each candidate stencil, a smoothness indicator is computed. This indicator quantifies how smooth the approximation is within that stencil, with lower values indicating smoother regions.
3. **Weight Calculation**: Nonlinear weights are computed based on the smoothness indicators. Stencils with lower smoothness indicators receive higher weights, allowing the scheme to adaptively favor smoother regions.
4. **Reconstruction**: The final high-order approximation is obtained by combining the candidate polynomials using the computed weights.
5. **Flux Evaluation**: The reconstructed values are used to approximate the variable of interest at the cell interfaces, which are then used in the numerical flux calculations.

In FiniteDiffWENO5.jl, two forms of advection operators are currently supported:
1. **Non-conservative Form** $\mathbf{v} \cdot \nabla u$, where the velocity field $\mathbf{v}$ and scalar field $u$ are both defined at the same grid locations (collocated grid), and
2. **Conservative Form** $\nabla \cdot (\mathbf{v}u)$, where $\mathbf{v}$ is defined on cell faces and $u$ at cell centers (staggered grid).

In both formulations, only the scalar field $u$ is reconstructed at the cell interfaces using the WENO scheme.

The package currently implements the WENO-Z reconstruction developed by [Borges et al. (2008)](https://doi.org/10.1016/j.jcp.2007.11.038). This variant introduces a modified computation of the nonlinear weights that improves accuracy near critical points—where the first derivative of the solution vanishes—while preserving the robust, non-oscillatory behavior of the classical WENO methods. Additional reconstruction variants may be included in future versions.

## Simplex-constrained material fractions

For material fractions, independent nonlinear weights generally break `Σₖϕₖ = 1`.
`MultiphaseWENOScheme` instead averages the phase smoothness indicators and uses the
resulting weights `ωᵣ` for every phase. Because each candidate reconstruction `sᵣ` is
exact for constants,

```math
\sum_k \phi_{k,f} = \sum_r \omega_r \sum_k s_r(\phi_k)
                    = \sum_r \omega_r s_r(1) = 1.
```

One common Zhang-Shu coefficient `θ` then limits the complete face vector toward its
donor composition. This convex combination preserves both the sum and the interval
`[0,1]`. Independent limiter coefficients would not preserve the sum.

For staggered velocity, material fractions obey

```math
\partial_t\phi_k + \nabla\!\cdot(\mathbf{v}\phi_k)
    = \phi_k\,\nabla\!\cdot\mathbf{v}.
```

The source uses the same two-point discrete divergence as the summed face fluxes, so
the two terms cancel algebraically for `Σₖϕₖ`, up to floating-point roundoff. The
collocated operator directly discretises the advective form and requires no divergence
source.
