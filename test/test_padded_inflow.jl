using Test
using FiniteDiffWENO5
using FiniteDiffWENO5: padded_weno_scheme, fill_physical_ghosts!, PaddedExtent,
    ProcessBC, scalar_operator_1D!, scalar_operator_2D!, scalar_operator_3D!,
    apply_lower_inflow!, apply_upper_inflow!, apply_x_lower_inflow!,
    apply_multiphase_x_lower_inflow!, apply_multiphase_lower_inflow!,
    multiphase_inflow_value, default_extent, WENOScheme

inflow_detfield(n) = [1.0 + 0.3sinpi(2 * (i - 0.5) / n) + 0.15cospi(4 * (i - 0.5) / n) for i in 1:n]
inflow_detfield(n, m) = [1.0 + 0.3sinpi(2 * (i - 0.5) / n) * cospi(2 * (j - 0.5) / m) for i in 1:n, j in 1:m]
inflow_detfield(n, m, p) = [1.0 + 0.2sinpi(2 * (i - 0.5) / n) * cospi(2 * (j - 0.5) / m) * sinpi(2 * (k - 0.5) / p) for i in 1:n, j in 1:m, k in 1:p]

@testset "inflow installation at the physical face" begin

    @testset "1D PrescribedInflowBC single-operator parity (both forms)" for form in (:nonconservative, :conservative)
        n = 16
        halo = 3
        inflow_lo, inflow_hi = 0.37, 1.91
        boundary = (PrescribedInflowBC(inflow_lo), PrescribedInflowBC(inflow_hi))
        u = inflow_detfield(n)
        vcell = (; x = inflow_detfield(n) .+ 0.5)

        weno_s = WENOScheme(copy(u); boundary, form, stag = false, multithreading = false)
        α_s = form === :conservative ? FiniteDiffWENO5.lf_speed(vcell.x) : 0.0
        du_s = zeros(n)
        scalar_operator_1D!(du_s, u, vcell, weno_s, n, inv(n), 0.0, 1.0, α_s)

        c0 = zeros(n + 2halo)
        c0[(halo + 1):(halo + n)] .= u
        extent = PaddedExtent{1}((n,), (halo,), (n,), (false,), :cell)
        fill_physical_ghosts!(c0, extent, boundary)

        vcell_p = zeros(n + 2halo)
        vcell_p[(halo + 1):(halo + n)] .= vcell.x
        fill_physical_ghosts!(vcell_p, extent, boundary)

        weno_p = padded_weno_scheme(zeros(n + 2halo), (halo,); boundary, form, stag = false, multithreading = false)
        α_p = form === :conservative ? FiniteDiffWENO5.lf_speed(view(vcell_p, (halo + 1):(halo + n))) : 0.0
        du_p = zeros(n + 2halo)
        scalar_operator_1D!(du_p, c0, (; x = vcell_p), weno_p, n + 2halo, inv(n), 0.0, 1.0, α_p)

        @test du_p[(halo + 1):(halo + n)] == du_s
    end

    @testset "2D PrescribedInflowBC single-operator parity with an array-valued inflow (nonconservative)" begin
        nx, ny = 14, 8
        halo = 3
        inflow_west = [0.5 + 0.1j for j in 1:ny] # array-valued: exercises the tangential offset
        boundary = (PrescribedInflowBC(inflow_west), ExtrapolateBC(), ExtrapolateBC(), ExtrapolateBC())
        u = inflow_detfield(nx, ny)
        vcell = (x = inflow_detfield(nx, ny) .+ 0.3, y = inflow_detfield(nx, ny) .+ 0.2)

        weno_s = WENOScheme(copy(u); boundary, form = :nonconservative, stag = false, multithreading = false)
        du_s = zeros(nx, ny)
        scalar_operator_2D!(du_s, u, vcell, weno_s, nx, ny, inv(nx), inv(ny), 0.0, 1.0, 0.0, 0.0)

        c0 = zeros(nx + 2halo, ny + 2halo)
        c0[(halo + 1):(halo + nx), (halo + 1):(halo + ny)] .= u
        extent = PaddedExtent{2}((nx, ny), (halo, halo), (nx, ny), (false, false), :cell)
        fill_physical_ghosts!(c0, extent, boundary)

        vcell_p = (
            x = zeros(nx + 2halo, ny + 2halo),
            y = zeros(nx + 2halo, ny + 2halo),
        )
        vcell_p.x[(halo + 1):(halo + nx), (halo + 1):(halo + ny)] .= vcell.x
        vcell_p.y[(halo + 1):(halo + nx), (halo + 1):(halo + ny)] .= vcell.y
        fill_physical_ghosts!(vcell_p.x, extent, boundary)
        fill_physical_ghosts!(vcell_p.y, extent, boundary)

        weno_p = padded_weno_scheme(zeros(nx + 2halo, ny + 2halo), (halo, halo); boundary, form = :nonconservative, stag = false, multithreading = false)
        du_p = zeros(nx + 2halo, ny + 2halo)
        scalar_operator_2D!(du_p, c0, vcell_p, weno_p, nx + 2halo, ny + 2halo, inv(nx), inv(ny), 0.0, 1.0, 0.0, 0.0)

        @test du_p[(halo + 1):(halo + nx), (halo + 1):(halo + ny)] == du_s
    end

    @testset "3D PrescribedInflowBC single-operator smoke test (nonconservative)" begin
        n = 8
        halo = 3
        boundary = (PrescribedInflowBC(0.8), ExtrapolateBC(), ExtrapolateBC(), ExtrapolateBC(), ExtrapolateBC(), ExtrapolateBC())
        u = inflow_detfield(n, n, n)
        vcell = (x = inflow_detfield(n, n, n) .+ 0.4, y = inflow_detfield(n, n, n) .+ 0.1, z = inflow_detfield(n, n, n) .+ 0.2)

        weno_s = WENOScheme(copy(u); boundary, form = :nonconservative, stag = false, multithreading = false)
        du_s = zeros(n, n, n)
        scalar_operator_3D!(du_s, u, vcell, weno_s, n, n, n, inv(n), inv(n), inv(n), 0.0, 1.0, 0.0, 0.0, 0.0)

        c0 = zeros(n + 2halo, n + 2halo, n + 2halo)
        c0[(halo + 1):(halo + n), (halo + 1):(halo + n), (halo + 1):(halo + n)] .= u
        extent = PaddedExtent{3}((n, n, n), (halo, halo, halo), (n, n, n), (false, false, false), :cell)
        fill_physical_ghosts!(c0, extent, boundary)

        vcell_p = (x = zeros(n + 2halo, n + 2halo, n + 2halo), y = zeros(n + 2halo, n + 2halo, n + 2halo), z = zeros(n + 2halo, n + 2halo, n + 2halo))
        r = (halo + 1):(halo + n)
        vcell_p.x[r, r, r] .= vcell.x
        vcell_p.y[r, r, r] .= vcell.y
        vcell_p.z[r, r, r] .= vcell.z
        fill_physical_ghosts!(vcell_p.x, extent, boundary)
        fill_physical_ghosts!(vcell_p.y, extent, boundary)
        fill_physical_ghosts!(vcell_p.z, extent, boundary)

        weno_p = padded_weno_scheme(zeros(n + 2halo, n + 2halo, n + 2halo), (halo, halo, halo); boundary, form = :nonconservative, stag = false, multithreading = false)
        npad = n + 2halo
        du_p = zeros(npad, npad, npad)
        scalar_operator_3D!(du_p, c0, vcell_p, weno_p, npad, npad, npad, inv(n), inv(n), inv(n), 0.0, 1.0, 0.0, 0.0, 0.0)

        @test du_p[r, r, r] ≈ du_s rtol = 0 atol = 1.0e-13
    end

    @testset "1D conservative PrescribedInflowBC exterior-state reads use the physical edge" begin
        # Direct check that `apply_conservative_inflow_1d!` reads u/v at the
        # PHYSICAL edge (halo+1 / halo+n), not the array edge (1 / npad),
        # which would read pad garbage instead of the true boundary state.
        n = 10
        halo = 3
        u = zeros(n + 2halo)
        u[(halo + 1):(halo + n)] .= inflow_detfield(n)
        u[1:halo] .= -999.0 # garbage pad — if the fix reads array-begin this leaks in
        u[(halo + n + 1):end] .= -999.0
        v = ones(n + 2halo)
        boundary = (PrescribedInflowBC(1.0), PrescribedInflowBC(2.0))
        extent = PaddedExtent{1}((n,), (halo,), (n,), (false,), :cell)

        fl = (; x = zeros(n + 2halo + 1))
        fr = (; x = zeros(n + 2halo + 1))
        α = 5.0
        FiniteDiffWENO5.apply_conservative_inflow_1d!(fl, fr, u, v, α, boundary, extent)

        expected_lo = FiniteDiffWENO5.lf_split_minus(u[halo + 1], v[halo + 1], α)
        expected_hi = FiniteDiffWENO5.lf_split_plus(u[halo + n], v[halo + n], α)
        @test fr.x[halo + 1] == expected_lo
        @test fl.x[halo + n + 1] == expected_hi
        # nothing written at the array edges themselves
        @test fl.x[1] == 0.0 && fr.x[1] == 0.0
        @test fl.x[end] == 0.0 && fr.x[end] == 0.0
    end

    @testset "ProcessBC is a silent no-op for scalar installers" begin
        n = 10
        halo = 3
        extent = PaddedExtent{1}((n,), (halo,), (n,), (false,), :cell)
        flux = fill(NaN, n + 2halo + 1)
        apply_lower_inflow!(flux, ProcessBC(), extent)
        apply_upper_inflow!(flux, ProcessBC(), extent)
        @test all(isnan, flux) # untouched
    end

    @testset "ProcessBC/tag 3 rejected as user input, resolved tuple accepted internally" begin
        c0 = zeros(10)
        @test_throws ArgumentError WENOScheme(c0; boundary = (ProcessBC(), ExtrapolateBC()), form = :nonconservative)
        @test_throws ArgumentError WENOScheme(c0; boundary = (3, ExtrapolateBC()), form = :nonconservative)
    end

    @testset "multiphase installers: physical face, owned tangential window, correct offset" begin
        nx, ny = 12, 6
        halo = 3
        NP = 2
        extent = PaddedExtent{2}((nx, ny), (halo, halo), (nx, ny), (false, false), :cell)
        inflow = (0.3, 0.7) # scalar composition (sums to 1)
        bc = PrescribedInflowBC(inflow)

        flux = ntuple(_ -> fill(NaN, nx + 2halo, ny + 2halo), NP)
        apply_multiphase_x_lower_inflow!(flux, bc, extent)

        face = halo + 1
        for k in 1:NP
            # written exactly on the owned tangential window at the physical face
            for j in (halo + 1):(halo + ny)
                @test flux[k][face, j] == multiphase_inflow_value(bc, k)
            end
            # nothing written outside the owned tangential window
            @test all(isnan, flux[k][face, 1:halo])
            @test all(isnan, flux[k][face, (halo + ny + 1):end])
            # nothing written off the physical face
            @test all(isnan, flux[k][1:(face - 1), :])
            @test all(isnan, flux[k][(face + 1):end, :])
        end
    end

    @testset "multiphase installers with an array-valued composition: correct tangential offset" begin
        nx, ny = 12, 6
        halo = 3
        NP = 2
        extent = PaddedExtent{2}((nx, ny), (halo, halo), (nx, ny), (false, false), :cell)
        profile = [0.2 + 0.05j for j in 1:ny] # owned-sized (ny entries), not padded-sized
        inflow = (profile, [1.0 - c for c in profile])
        bc = PrescribedInflowBC(inflow)

        flux = ntuple(_ -> fill(NaN, nx + 2halo, ny + 2halo), NP)
        apply_multiphase_x_lower_inflow!(flux, bc, extent)

        face = halo + 1
        for (jlocal, j) in enumerate((halo + 1):(halo + ny))
            @test flux[1][face, j] == profile[jlocal]
            @test flux[2][face, j] == 1.0 - profile[jlocal]
        end
    end

    @testset "ProcessBC is a silent no-op for multiphase installers" begin
        nx, ny = 12, 6
        halo = 3
        NP = 2
        extent = PaddedExtent{2}((nx, ny), (halo, halo), (nx, ny), (false, false), :cell)
        flux = ntuple(_ -> fill(NaN, nx + 2halo, ny + 2halo), NP)
        apply_multiphase_x_lower_inflow!(flux, ProcessBC(), extent)
        apply_multiphase_lower_inflow!((fill(NaN, 3),), ProcessBC(), PaddedExtent{1}((3,), (0,), (3,), (false,), :cell))
        @test all(x -> all(isnan, x), flux)
    end

    @testset "--check-bounds friendliness: installer never indexes outside a padded flux/value array" begin
        # A tangential array smaller than the padded extent — if any installer
        # looped the padded (not owned) tangential range, this would read out
        # of bounds under `--check-bounds=yes`.
        nx, ny = 10, 5
        halo = 3
        extent = PaddedExtent{2}((nx, ny), (halo, halo), (nx, ny), (false, false), :cell)
        profile = collect(1.0:ny) # exactly owned-sized, no slack for an over-run
        bc = PrescribedInflowBC(profile)
        flux = zeros(nx + 2halo, ny + 2halo)
        @test apply_x_lower_inflow!(flux, bc, extent) === nothing
    end

    @testset "Existing serial inflow behaviour is unaffected (unpadded scheme)" begin
        n = 10
        boundary = (PrescribedInflowBC(1.5), ExtrapolateBC())
        weno = WENOScheme(zeros(n); boundary, form = :nonconservative, multithreading = false)
        u = inflow_detfield(n)
        vcell = (; x = ones(n))
        du = zeros(n)
        scalar_operator_1D!(du, u, vcell, weno, n, inv(n), 0.0, 1.0, 0.0)
        # sanity: the operator ran without needing padding-specific arguments
        @test all(isfinite, du)
    end
end
