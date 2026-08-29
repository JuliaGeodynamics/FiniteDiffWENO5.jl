# [Background](@id background)

The weighted essentially non-oscillatory (WENO) schemes form a class of high-order accurate numerical methods for solving hyperbolic partial differential equations (PDEs). They are particularly effective at resolving sharp gradients and discontinuities without introducing spurious oscillations. The WENO schemes achieve this by constructing nonlinear convex combinations of lower-order candidate polynomials, where the weights are determined by the local smoothness of the solution. This approach allows the method to retain high-order accuracy in smooth regions while automatically reducing to lower-order, more stable behavior near discontinuities. The WENO framework was first introduced by [Liu et al. 1994](https://doi.org/10.1006/jcph.1994.1187]), building upon the earlier essentially non-oscillatory (ENO) schemes developed by [Harten et al. 1987](https://doi.org/10.1016/0021-9991(87)90031-3).

In practice, WENO schemes can be formulated either in a finite-volume or finite-difference framework. In this package, we adopt the finite-difference formulation, which is particularly well-suited for problems defined on structured grids due to its simplicity and computational efficiency. The time integration is based on a third order strong stability preserving Runge-Kutta (SSP-RK3) method.

The implementation of a finite-difference WENO scheme involves the following main steps:

2. **Smoothness Indicators**: For each candidate stencil, a smoothness indicator is computed. This indicator quantifies how smooth the approximation is within that stencil, with lower values indicating smoother regions.
3. **Weight Calculation**: Nonlinear weights are computed based on the smoothness indicators. Stencils with lower smoothness indicators receive higher weights, allowing the scheme to adaptively favor smoother regions.
4. **Reconstruction**: The final high-order approximation is obtained by combining the candidate polynomials using the computed weights.
5. **Flux Evaluation**: The reconstructed values are used to approximate the variable of interest at the cell interfaces, which are then used in the numerical flux calculations.

The package currently implements the WENO-Z reconstruction developed by [Borges et al. (2008)](https://doi.org/10.1016/j.jcp.2007.11.038). This variant introduces a modified computation of the nonlinear weights that improves accuracy near critical points—where the first derivative of the solution vanishes—while preserving the robust, non-oscillatory behavior of the classical WENO methods. Additional reconstruction variants may be included in future versions.

## PDE form and velocity layout are independent choices

`WENOScheme` takes two keywords that answer two different questions:

- `form`: which equation is being solved,
  - `:conservative`: $\partial_t u + \nabla\cdot(\mathbf{v}u) = 0$,
  - `:nonconservative`: $\partial_t u + \mathbf{v}\cdot\nabla u = 0$;
- `stag`: where the velocity lives,
  - `false`: $\mathbf{v}$ is collocated with $u$ (cell centers),
  - `true`: $\mathbf{v}$ is face-staggered (the normal component of $\mathbf{v}$ lives on the faces normal to it).

All four combinations are supported and are all fifth order accurate in smooth regions.

### Conservative form: Lax-Friedrichs split-flux reconstruction

For `form=:conservative`, the point flux $f = vu$ is formed directly at the locations where $u$ lives, then split globally,
```math
f^+ = \tfrac12(f + \alpha u), \qquad f^- = \tfrac12(f - \alpha u), \qquad \alpha = \max|v|,
```
so that $\partial f^+/\partial u \ge 0$ and $\partial f^-/\partial u \le 0$. $f^+$ is reconstructed with the upwind-biased WENO operator and $f^-$ with the downwind-biased one, and the numerical flux at each face is their sum; the semi-discrete update differences these face fluxes. This is the standard global Lax-Friedrichs flux-splitting construction, and it is genuinely fifth order because it reconstructs the flux function itself rather than reconstructing $u$ and multiplying afterward.

### Non-conservative (material) form

For `form=:nonconservative`, the directional derivative of $u$ is reconstructed with the same one-sided WENO operators used for the conservative flux, then multiplied by the velocity at the same location. No splitting is needed here because there is no flux to conserve.

### Bridging a staggered velocity

Both forms need $u$ and $v$ available at the same location. When `stag=true`, the face-normal velocity component is first mapped to cell centers with a fifth-order ENO5 point interpolation (`eno5_face_to_center!` and its multi-dimensional wrappers), following [Mishra, Parés-Pulido & Pressel (2020)](https://arxiv.org/abs/1905.13665), Algorithm 2. This interpolation is done once per `WENO_step!` call, outside the three SSP-RK3 stages, since the velocity does not change across them. Both `form` values then operate on this cell-centered velocity exactly as they would for a collocated field. The flux-splitting and material-derivative constructions above are unaware of whether the velocity was originally staggered.

## Simplex-constrained material fractions

`MultiphaseWENOScheme` solves the material (non-conservative) transport equation for every phase fraction simultaneously,
```math
\partial_t\phi_k + \mathbf{v}\cdot\nabla\phi_k = 0, \qquad k = 1,\dots,N_P,
```
subject to $\phi_k \in [0,1]$ and $\sum_k \phi_k = 1$ at every point. There is no `form` keyword for this scheme. The simplex constraint only has a well-defined meaning for material transport, since $\phi_k$ are fractions of a whole, not conserved densities in the flux sense.

For material fractions, independent nonlinear weights per phase generally break $\sum_k\phi_k = 1$.
`MultiphaseWENOScheme` instead averages the phase smoothness indicators and uses the
resulting weights $\omega_r$ for every phase. Because each candidate reconstruction $s_r$ is
exact for constants,

```math
\sum_k \phi_{k,f} = \sum_r \omega_r \sum_k s_r(\phi_k)
                    = \sum_r \omega_r s_r(1) = 1.
```

One common Zhang-Shu coefficient $\theta$ then limits the complete face vector toward its
donor composition. This convex combination preserves both the sum and the interval
$[0,1]$. Independent limiter coefficients would not preserve the sum.

The same argument extends to the semi-discrete update: because $\sum_k \phi_{k,\text{face}} = 1$ at every face regardless of which cell-centered velocity multiplies it, summing the material-derivative update over $k$ gives
```math
\sum_k \partial_t\phi_k = v\cdot\Big(\sum_k \hat\phi_{k,\text{right}} - \sum_k \hat\phi_{k,\text{left}}\Big)/\Delta x = v\cdot(1-1)/\Delta x = 0
```
exactly. No divergence source or bookkeeping term is needed on either the collocated or the staggered path. On the staggered path the velocity is prepared to cell centers by the same fifth-order ENO5 interpolation described above; because the multiphase update reuses the same shared-weight cancellation independent of that interpolation's accuracy, the simplex invariant holds at roundoff regardless of the velocity layout, while the *rate of convergence* to the true solution is fifth order precisely because the ENO5 bridging step is itself fifth order.

