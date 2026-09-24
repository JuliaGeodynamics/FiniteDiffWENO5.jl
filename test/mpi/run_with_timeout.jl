# Launcher-level deadlock timeout: runs the given command as a subprocess
# and kills it if it hasn't finished within TIMEOUT_SECONDS, turning a
# genuine MPI deadlock into a clear failure instead of a hung CI job.
#
# Usage: julia run_with_timeout.jl TIMEOUT_SECONDS -- CMD [ARGS...]

function main(args)
    sep = findfirst(==("--"), args)
    (sep === nothing || sep == length(args)) && error(
        "usage: julia run_with_timeout.jl TIMEOUT_SECONDS -- CMD [ARGS...]"
    )
    timeout = parse(Float64, args[1])
    cmd = Cmd(String.(args[(sep + 1):end]))

    proc = run(pipeline(cmd; stdout = stdout, stderr = stderr); wait = false)
    timed_out = Ref(false)
    timer = Timer(timeout) do _
        if process_running(proc)
            timed_out[] = true
            @warn "run_with_timeout: command exceeded $(timeout)s — killing (likely deadlock)"
            kill(proc, Base.SIGKILL)
        end
    end
    wait(proc)
    close(timer)

    if timed_out[]
        println(stderr, "run_with_timeout: TIMED OUT after $(timeout)s")
        exit(124) # conventional timeout exit code
    end
    exit(proc.exitcode)
end

main(ARGS)
