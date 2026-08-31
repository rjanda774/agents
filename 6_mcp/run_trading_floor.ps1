# Auto-restart wrapper for trading_floor.py.
#
# trading_floor.py's own scheduler loop is a `while True` that's supposed to run
# forever -- it should never exit on its own. If it does exit (any exit code,
# including 0), something killed the process out from under it: the machine woke
# from sleep into a broken state, `uv`/Python itself crashed hard, an unhandled
# exception slipped past every guard in the code, etc. This wrapper's only job is
# to notice that and start it right back up, logging when/why so the cause is
# diagnosable afterward instead of just "it's been silent since Saturday."
#
# This does NOT fix the machine going to sleep -- nothing runs while the OS is
# suspended, wrapper included. Pair this with disabling sleep (Settings > System >
# Power & sleep > Sleep: Never while plugged in, or `powercfg /change
# standby-timeout-ac 0`) and, if you want it running even when you're logged out,
# a Task Scheduler entry pointed at this script (trigger: At log on / At startup;
# Settings: restart on failure) rather than a manually-opened terminal window.
#
# Usage: powershell -ExecutionPolicy Bypass -File run_trading_floor.ps1
# Stop it with Ctrl+C (propagates to the child `uv run` process too).

$ErrorActionPreference = "Continue"

# Always run from this script's own directory, regardless of where/how it's
# launched from (a raw terminal vs. Task Scheduler's default working directory).
Set-Location -Path $PSScriptRoot

$logDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$logFile = Join-Path $logDir "trading_floor_wrapper.log"

$RestartDelaySeconds = 10

function Write-Log($message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $message
    Write-Host $line
    Add-Content -Path $logFile -Value $line
}

Write-Log "Wrapper started (PID $PID). Logging restarts to $logFile"

while ($true) {
    Write-Log "Starting: uv run trading_floor.py"
    # -Wait blocks until the child exits; -PassThru gets us its exit code.
    # Not redirecting stdout/stderr here -- trading_floor.py's own prints (cycle
    # logs, tracebacks from the guards already in the code) still go straight to
    # this console/window, same as running it directly.
    $proc = Start-Process -FilePath "uv" -ArgumentList "run", "trading_floor.py" `
        -NoNewWindow -PassThru -Wait
    $exitCode = $proc.ExitCode

    Write-Log "trading_floor.py exited with code $exitCode -- it should run forever, so this means something killed it (sleep/resume, a crash, etc). Restarting in $RestartDelaySeconds s..."
    Start-Sleep -Seconds $RestartDelaySeconds
}
