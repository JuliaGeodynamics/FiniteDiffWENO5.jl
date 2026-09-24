using Test
using FiniteDiffWENO5
using FiniteDiffWENO5: padded_weno_scheme, fill_physical_ghosts!, PaddedExtent,
    ProcessBC, left_index, right_index, valid_boundary, validate_boundary,
    validate_multiphase_boundary, eno5_difference_valid, eno5_stencil_start,
    eno5_face_to_center!, prepare_velocity!, scalar_operator_1D!,
    scalar_operator_2D!, scalar_operator_3D!, lf_speed, WENOScheme

# ------------------------------------------------------------------
# Small deterministic (non-trivial, non-symmetric) test fields so a bug
# that only shows up off-center or off-smooth is not accidentally masked.
# ------------------------------------------------------------------
detfield(n) = [1.0 + 0.3sinpi(2 * (i - 0.5) / n) + 0.15cospi(4 * (i - 0.5) / n) for i in 1:n]
detfield(n, m) = [1.0 + 0.3sinpi(2 * (i - 0.5) / n) * cospi(2 * (j - 0.5) / m) for i in 1:n, j in 1:m]
detfield(n, m, p) = [1.0 + 0.2sinpi(2 * (i - 0.5) / n) * cospi(2 * (j - 0.5) / m) * sinpi(2 * (k - 0.5) / p) for i in 1:n, j in 1:m, k in 1:p]
detface(n) = [1.0 + 0.4sinpi(2 * i / n) for i in 0:n]

@testset "padding-aware boundaries" begin

    @testset "ProcessBC index methods match ExtrapolateBC" begin
        for i in 1:10, d in 0:3, nx in (10,)
            @test left_index(i, d, nx, ProcessBC()) == left_index(i, d, nx, ExtrapolateBC())
            @test right_index(i, d, nx, ProcessBC()) == right_index(i, d, nx, ExtrapolateBC())
        end
    end

    @testset "ProcessBC and integer tag 3 rejected by user validators" begin
        @test valid_boundary(ProcessBC()) == false
        @test valid_boundary(3) == false
        @test_throws ArgumentError validate_boundary((ProcessBC(), ExtrapolateBC()), 1)
        @test_throws ArgumentError validate_boundary((3, ExtrapolateBC()), 1)
        @test_throws ArgumentError validate_multiphase_boundary(
            (ProcessBC(), ExtrapolateBC()), 1, (5,), 2, Float64
        )
        u = zeros(6)
        @test_throws ArgumentError WENOScheme(u; boundary = (ProcessBC(), ExtrapolateBC()), form = :nonconservative)
    end

    @testset "kernel-level Integer tag 3 clamps, does not alias periodic" begin
        nx = 10
        # tag 3 must NOT wrap like periodic (tag other-than-0/1, today's `else`
        # branch): at i=1, d=3, periodic would give mod1(1-3,10)=8; the clamp
        # must give 1.
        @test left_index(1, 3, nx, 3) == 1
        @test right_index(nx, 3, nx, 3) == nx
        # and it must actually clamp like Extrapolate/Neumann (tag 1), not
        # like Dirichlet (tag 0) either.
        @test left_index(1, 3, nx, 3) == left_index(1, 3, nx, 1)
        @test right_index(nx, 3, nx, 3) == right_index(nx, 3, nx, 1)
    end

    @testset "padded_weno_scheme: construction and size assertion" begin
        n = 12
        halo = (3,)
        c0 = zeros(n + 2halo[1])
        weno = padded_weno_scheme(
            c0, halo; boundary = (ExtrapolateBC(), ExtrapolateBC()), form = :nonconservative,
        )
        @test size(weno.du) == size(c0)
        @test size(weno.ut) == size(c0)

        bad = zeros(2halo[1] - 1) # too small: owned extent would be negative
        @test_throws Exception padded_weno_scheme(
            bad, halo; boundary = (ExtrapolateBC(), ExtrapolateBC()), form = :nonconservative,
        )

        # A resolved tuple containing ProcessBC must be ACCEPTED here (unlike
        # the user-facing validators above).
        weno_proc = padded_weno_scheme(
            c0, halo; boundary = (ProcessBC(), ProcessBC()), form = :nonconservative,
        )
        @test weno_proc.boundary[1] isa ProcessBC
    end

    @testset "unpadded WENOScheme construction is byte-identical to today" begin
        n = 16
        c0 = detfield(n)
        w1 = WENOScheme(copy(c0); boundary = (ExtrapolateBC(), ExtrapolateBC()), form = :nonconservative, stag = false)
        w2 = WENOScheme(copy(c0); boundary = (ExtrapolateBC(), ExtrapolateBC()), form = :nonconservative, stag = false)
        @test w1.du == w2.du
        @test size(w1.du) == (n,)
        @test w1.fl.x == w2.fl.x
    end

    # Compare padded and unpadded operators at extrapolated and periodic faces.

    function parity_1d(boundary_kind::Symbol, stag::Bool, form::Symbol)
        n = 20
        halo = 3
        u = detfield(n)
        vface = detface(n)

        boundary = boundary_kind === :extrapolate ? (ExtrapolateBC(), ExtrapolateBC()) :
            (PeriodicBC(), PeriodicBC())

        # --- serial (unpadded) reference ---
        weno_s = WENOScheme(copy(u); boundary, form, stag, multithreading = false)
        vcell_s = if stag
            vc = zeros(n)
            FiniteDiffWENO5.eno5_face_to_center!(vc, vface; periodic = boundary_kind === :periodic)
            (; x = vc)
        else
            (; x = detfield(n)) # collocated velocity, arbitrary but deterministic
        end
        α_s = form === :conservative ? FiniteDiffWENO5.lf_speed(vcell_s.x) : 0.0
        du_s = zeros(n)
        scalar_operator_1D!(du_s, u, vcell_s, weno_s, n, inv(n), 0.0, 1.0, α_s)

        # --- padded counterpart ---
        c0 = zeros(n + 2halo)
        c0[(halo + 1):(halo + n)] .= u
        extent = PaddedExtent{1}((n,), (halo,), (n,), (boundary_kind === :periodic,), :cell)
        fill_physical_ghosts!(c0, extent, boundary)

        weno_p = padded_weno_scheme(zeros(n + 2halo), (halo,); boundary, form, stag, multithreading = false)
        u_p = copy(c0)

        vcell_p = if stag
            vface_p = zeros(n + 2halo + 1)
            vface_p[(halo + 1):(halo + n + 1)] .= vface
            fill_physical_ghosts!((; x = vface_p), extent, boundary)
            prepare_velocity!(weno_p, (; x = vface_p))
            fill_physical_ghosts!(weno_p.vcenter, extent, boundary)
            weno_p.vcenter
        else
            vc_p = zeros(n + 2halo)
            vc_p[(halo + 1):(halo + n)] .= vcell_s.x
            fill_physical_ghosts!(vc_p, extent, boundary)
            (; x = vc_p)
        end
        α_p = form === :conservative ? FiniteDiffWENO5.lf_speed(view(vcell_p.x, (halo + 1):(halo + n))) : 0.0
        du_p = zeros(n + 2halo)
        scalar_operator_1D!(du_p, u_p, vcell_p, weno_p, n + 2halo, inv(n), 0.0, 1.0, α_p)

        @test du_p[(halo + 1):(halo + n)] == du_s
        if stag
            @test weno_p.vcenter.x[(halo + 1):(halo + n)] == vcell_s.x
        end
    end

    @testset "1D single-operator parity: $bk, stag=$stag, $form" for bk in (:extrapolate, :periodic),
            stag in (false, true), form in (:nonconservative, :conservative)

        parity_1d(bk, stag, form)
    end

    @testset "1D: eno5_difference_valid/eno5_stencil_start match serial at every owned cell (ExtrapolateBC, padded)" begin
        n = 16
        halo = 3
        vface_serial = detface(n)
        vface_padded = zeros(n + 2halo + 1)
        vface_padded[(halo + 1):(halo + n + 1)] .= vface_serial
        extent = PaddedExtent{1}((n,), (halo,), (n,), (false,), :cell)
        fill_physical_ghosts!((; x = vface_padded), extent, (ExtrapolateBC(), ExtrapolateBC()))

        restriction = FiniteDiffWENO5.ENO5PhysicalRestriction(halo + 1, halo + n, halo + 1, halo + n + 1, true, true)
        npad = n + 2halo
        for i in 1:n
            s_serial = eno5_stencil_start(vface_serial, i, n, false)
            s_padded = eno5_stencil_start(vface_padded, i + halo, npad, false, restriction)
            @test s_padded - halo == s_serial
        end
    end

    @testset "1D thin global axis pins ENO5-vs-linear selection to global extent" begin
        for n in (3, 4), periodic in (false, true)
            n == 3 && periodic && continue # eno5 periodic needs >= 5 cells regardless; linear path only
            halo = 3
            u_serial = collect(1.0:n)
            vface_serial = periodic ? [1.0 + 0.1i for i in 0:(n - 1)] : [1.0 + 0.1i for i in 0:n]
            boundary = periodic ? (PeriodicBC(), PeriodicBC()) : (ExtrapolateBC(), ExtrapolateBC())

            vcenter_serial = zeros(n)
            FiniteDiffWENO5.face_to_center_direction!(vcenter_serial, vface_serial, 1; periodic)

            c0 = zeros(n + 2halo)
            c0[(halo + 1):(halo + n)] .= u_serial
            extent = PaddedExtent{1}((n,), (halo,), (n,), (periodic,), :cell)
            fill_physical_ghosts!(c0, extent, boundary)

            weno_p = padded_weno_scheme(zeros(n + 2halo), (halo,); boundary, form = :nonconservative, stag = true, multithreading = false)

            # A padded face array always carries the full n_phys+1 physical
            # faces regardless of periodicity — padding always forces
            # `periodic = false` for interpolation, so the array follows the nonperiodic
            # convention; for a periodic boundary the duplicate n_phys+1'th
            # face is left for `fill_physical_ghosts!` to write.
            vface_padded = zeros(n + 2halo + 1)
            if periodic
                vface_padded[(halo + 1):(halo + n)] .= vface_serial # n_phys+1'th face is the duplicate; fill_physical_ghosts! writes it
            else
                vface_padded[(halo + 1):(halo + n + 1)] .= vface_serial # genuine extra physical face
            end
            fill_physical_ghosts!((; x = vface_padded), extent, boundary)

            prepare_velocity!(weno_p, (; x = vface_padded))

            @test weno_p.vcenter.x[(halo + 1):(halo + n)] == vcenter_serial
        end
    end

    @testset "ProcessBC reproduces the interior of a larger serial run bit-for-bit" begin
        n_large = 40
        n_owned = 20
        halo = 3
        offset = 10 # owned window = large indices (offset+1):(offset+n_owned)

        u_large = detfield(n_large)
        vface_large = detface(n_large)
        boundary_large = (ExtrapolateBC(), ExtrapolateBC())

        weno_large = WENOScheme(copy(u_large); boundary = boundary_large, form = :nonconservative, stag = false, multithreading = false)
        vcell_large = (; x = detfield(n_large))
        du_large = zeros(n_large)
        scalar_operator_1D!(du_large, u_large, vcell_large, weno_large, n_large, inv(n_large), 0.0, 1.0, 0.0)

        u_p = zeros(n_owned + 2halo)
        u_p .= u_large[(offset - halo + 1):(offset + n_owned + halo)]
        vcell_p = (; x = vcell_large.x[(offset - halo + 1):(offset + n_owned + halo)])

        weno_p = padded_weno_scheme(
            zeros(n_owned + 2halo), (halo,); boundary = (ProcessBC(), ProcessBC()),
            form = :nonconservative, stag = false, multithreading = false,
        )
        du_p = zeros(n_owned + 2halo)
        scalar_operator_1D!(du_p, u_p, vcell_p, weno_p, n_owned + 2halo, inv(n_large), 0.0, 1.0, 0.0)

        @test du_p[(halo + 1):(halo + n_owned)] == du_large[(offset + 1):(offset + n_owned)]
    end

    @testset "2D mixed tags across axes: ProcessBC on x, ExtrapolateBC on y" begin
        nx_large = 40
        ny = 12
        nx_owned = 20
        halo = 3
        offset = 10

        u_large = detfield(nx_large, ny)
        boundary_large = (ExtrapolateBC(), ExtrapolateBC(), ExtrapolateBC(), ExtrapolateBC())
        weno_large = WENOScheme(copy(u_large); boundary = boundary_large, form = :nonconservative, stag = false, multithreading = false)
        vcell_large = (; x = detfield(nx_large, ny), y = detfield(nx_large, ny))
        du_large = zeros(nx_large, ny)
        scalar_operator_2D!(du_large, u_large, vcell_large, weno_large, nx_large, ny, inv(nx_large), inv(ny), 0.0, 1.0, 0.0, 0.0)

        u_p = u_large[(offset - halo + 1):(offset + nx_owned + halo), :]
        vcell_p = (
            x = vcell_large.x[(offset - halo + 1):(offset + nx_owned + halo), :],
            y = vcell_large.y[(offset - halo + 1):(offset + nx_owned + halo), :],
        )

        boundary_p = (ProcessBC(), ProcessBC(), ExtrapolateBC(), ExtrapolateBC())
        weno_p = padded_weno_scheme(
            zeros(nx_owned + 2halo, ny), (halo, 0); boundary = boundary_p,
            form = :nonconservative, stag = false, multithreading = false,
        )
        du_p = zeros(nx_owned + 2halo, ny)
        scalar_operator_2D!(du_p, u_p, vcell_p, weno_p, nx_owned + 2halo, ny, inv(nx_large), inv(ny), 0.0, 1.0, 0.0, 0.0)

        @test du_p[(halo + 1):(halo + nx_owned), :] == du_large[(offset + 1):(offset + nx_owned), :]
    end

    @testset "3D single-operator parity smoke test (ExtrapolateBC, stag=true)" begin
        n = 10
        halo = 3
        u = detfield(n, n, n)
        boundary = ntuple(_ -> ExtrapolateBC(), 6)

        weno_s = WENOScheme(copy(u); boundary, form = :nonconservative, stag = true, multithreading = false)
        vfx = [1.0 + 0.3sinpi(2i / n) for i in 0:n, j in 1:n, k in 1:n]
        vfy = [0.5 + 0.2cospi(2j / n) for i in 1:n, j in 0:n, k in 1:n]
        vfz = [0.4 + 0.1sinpi(2k / n) for i in 1:n, j in 1:n, k in 0:n]
        vcenter_s = (x = zeros(n, n, n), y = zeros(n, n, n), z = zeros(n, n, n))
        FiniteDiffWENO5.eno5_face_to_center!(vcenter_s, (; x = vfx, y = vfy, z = vfz); periodic = (; x = false, y = false, z = false))
        du_s = zeros(n, n, n)
        scalar_operator_3D!(du_s, u, vcenter_s, weno_s, n, n, n, inv(n), inv(n), inv(n), 0.0, 1.0, 0.0, 0.0, 0.0)

        c0 = zeros(n + 2halo, n + 2halo, n + 2halo)
        c0[(halo + 1):(halo + n), (halo + 1):(halo + n), (halo + 1):(halo + n)] .= u
        extent = PaddedExtent{3}((n, n, n), (halo, halo, halo), (n, n, n), (false, false, false), :cell)
        fill_physical_ghosts!(c0, extent, boundary)

        weno_p = padded_weno_scheme(
            zeros(n + 2halo, n + 2halo, n + 2halo), (halo, halo, halo); boundary, form = :nonconservative, stag = true, multithreading = false,
        )

        vfx_p = zeros(n + 2halo + 1, n + 2halo, n + 2halo)
        vfy_p = zeros(n + 2halo, n + 2halo + 1, n + 2halo)
        vfz_p = zeros(n + 2halo, n + 2halo, n + 2halo + 1)
        vfx_p[(halo + 1):(halo + n + 1), (halo + 1):(halo + n), (halo + 1):(halo + n)] .= vfx
        vfy_p[(halo + 1):(halo + n), (halo + 1):(halo + n + 1), (halo + 1):(halo + n)] .= vfy
        vfz_p[(halo + 1):(halo + n), (halo + 1):(halo + n), (halo + 1):(halo + n + 1)] .= vfz
        vface_p = (; x = vfx_p, y = vfy_p, z = vfz_p)
        fill_physical_ghosts!(vface_p, extent, boundary)
        prepare_velocity!(weno_p, vface_p)
        fill_physical_ghosts!(weno_p.vcenter, extent, boundary)

        npad = n + 2halo
        du_p = zeros(npad, npad, npad)
        scalar_operator_3D!(du_p, c0, weno_p.vcenter, weno_p, npad, npad, npad, inv(n), inv(n), inv(n), 0.0, 1.0, 0.0, 0.0, 0.0)

        r = (halo + 1):(halo + n)
        # `isapprox` with an extremely tight tolerance, not `==`: under
        # `--check-bounds=yes` recompilation, LLVM/SIMD codegen for this
        # kernel reorders the floating-point sum by ~1-2 ULP (confirmed
        # exactly 0 under normal compilation); a real bug would be many
        # orders of magnitude larger than this tolerance.
        @test du_p[r, r, r] ≈ du_s rtol = 0 atol = 1.0e-13
    end

    @testset "PrescribedInflowBC: fill_physical_ghosts! writes edge value, not prescribed value" begin
        n = 12
        halo = 3
        inflow = 99.0 # deliberately far from anything in u so a leak is obvious
        boundary = (PrescribedInflowBC(inflow), ExtrapolateBC())
        u = detfield(n)
        c0 = zeros(n + 2halo)
        c0[(halo + 1):(halo + n)] .= u
        extent = PaddedExtent{1}((n,), (halo,), (n,), (false,), :cell)
        fill_physical_ghosts!(c0, extent, boundary)

        # Every low-side ghost must equal the physical edge value (u[1]), never `inflow`.
        for g in 1:halo
            @test c0[g] == u[1]
            @test c0[g] != inflow
        end

        # Also correct on `ut`-shaped and on a velocity component (units-free check:
        # different array, same mechanism must apply identically).
        ut = zeros(n + 2halo)
        ut[(halo + 1):(halo + n)] .= detfield(n) .+ 5.0
        fill_physical_ghosts!(ut, extent, boundary)
        for g in 1:halo
            @test ut[g] == ut[halo + 1]
        end

        vface = zeros(n + 2halo + 1)
        vface[(halo + 1):(halo + n + 1)] .= detface(n)
        fill_physical_ghosts!((; x = vface), extent, boundary)
        for g in 1:halo
            @test vface[g] == vface[halo + 1]
        end
    end

    @testset "PrescribedInflowBC: scheme constructs and steps without crashing under padding" begin
        n = 10
        halo = 3
        boundary = (PrescribedInflowBC(1.0), ExtrapolateBC())
        weno_p = padded_weno_scheme(
            zeros(n + 2halo), (halo,); boundary, form = :nonconservative,
            stag = false, multithreading = false,
        )
        u_p = zeros(n + 2halo)
        u_p[(halo + 1):(halo + n)] .= detfield(n)
        extent = PaddedExtent{1}((n,), (halo,), (n,), (false,), :cell)
        fill_physical_ghosts!(u_p, extent, boundary)
        vcell_p = (; x = zeros(n + 2halo))
        du_p = zeros(n + 2halo)
        @test scalar_operator_1D!(du_p, u_p, vcell_p, weno_p, n + 2halo, inv(n), 0.0, 1.0, 0.0) === nothing
    end

    @testset "--check-bounds friendliness: index functions stay in range across the padded extent" begin
        n = 10
        halo = 3
        npad = n + 2halo
        for boundary in ((ExtrapolateBC(), ExtrapolateBC()), (PeriodicBC(), PeriodicBC()), (ProcessBC(), ProcessBC()))
            for i in 1:npad, d in 0:3
                li = left_index(i, d, npad, boundary[1])
                ri = right_index(i, d, npad, boundary[2])
                @test 1 <= li <= npad
                @test 1 <= ri <= npad
            end
        end
    end
end

@testset "Periodic process indexing preserves conservative reconstruction" begin
    n, h = 40, 3
    u = [1 + 0.3sinpi(2 * (i - 0.5) / n) + 0.15cospi(4 * (i - 0.5) / n) for i in 1:n]
    v = u .+ 0.5
    serial = WENOScheme(
        u; boundary = (PeriodicBC(), PeriodicBC()),
        form = :conservative, multithreading = false
    )
    process = ProcessBC(PeriodicBC())
    padded = FiniteDiffWENO5.padded_weno_scheme(
        zeros(n + 2h), (h,);
        boundary = (process, process), form = :conservative, multithreading = false
    )
    up = [u[mod1(i - h, n)] for i in 1:(n + 2h)]
    vp = [v[mod1(i - h, n)] for i in 1:(n + 2h)]
    alpha = FiniteDiffWENO5.lf_speed(v)
    FiniteDiffWENO5.conservative_semi_discretisation_weno5!(
        serial.du, u, (; x = v), serial, n, Float64(n), alpha
    )
    FiniteDiffWENO5.conservative_semi_discretisation_weno5!(
        padded.du, up, (; x = vp), padded, n + 2h, Float64(n), alpha
    )
    @test serial.fl.x == padded.fl.x[(h + 1):(h + n + 1)]
    @test serial.fr.x == padded.fr.x[(h + 1):(h + n + 1)]
    @test serial.du == padded.du[(h + 1):(h + n)]
    @test !valid_boundary(process)
    for i in 1:(n + 1), d in 0:3
        @test 1 <= left_index(i, d, n, process) <= n
        @test 1 <= right_index(i, d, n, process) <= n
    end
end
