<#
.SYNOPSIS
  Windows wrapper for the Cathie trading floor scheduler (trading_floor.py).

.DESCRIPTION
  trading_floor.py's own scheduler loop is meant to run forever (see CLAUDE.md:
  "starts the scheduler: runs every trader once, then sleeps RUN_EVERY_N_MINUTES
  and repeats"), so any exit from it -- clean or not -- during normal unattended
  operation is unexpected. This wrapper runs it under `uv run`, logs timestamped
  start/crash/restart events to logs\trading_floor_wrapper.log (appended across
  restarts, so history survives), and auto-restarts it up to $MaxRestarts times
  if it exits with a nonzero code before giving up.

  trading_floor.py's own stdout/stderr (including its own "Starting scheduler to
  run every N minutes" line) pass straight through to this console as usual --
  this wrapper only logs its OWN start/crash/restart events to the log file, it
  doesn't re-capture the child process's output into that file too.

  Recreated from observed behavior (this script was never checked into the repo
  before and got lost locally) -- adjust $MaxRestarts/$RestartDelaySeconds below
  if your original had different values.

.NOTES
  A manual Ctrl+C in an interactive console terminates this whole script (not
  just the inner `uv run` call) under PowerShell's default behavior, so stopping
  it by hand does NOT count as a crash and will not trigger a retry.
#>

$ErrorActionPreference = "Stop"

# Always run relative to this script's own folder (6_mcp), regardless of where
# it's invoked from.
Set-Location -Path $PSScriptRoot

$LogDir = Join-Path $PSScriptRoot "logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}
$LogFile = Join-Path $LogDir "trading_floor_wrapper.log"

$MaxRestarts = 3          # total retry ATTEMPTS after a crash (so up to 4 runs total: 1 initial + 3 retries)
$RestartDelaySeconds = 30 # pause between a crash and the next restart attempt

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    # Tee to both the console and the persistent log file.
    $line | Tee-Object -FilePath $LogFile -Append
}

Write-Log "Wrapper started (PID $PID). Logging restarts to $LogFile"

$failureCount = 0
while ($true) {
    Write-Log "Starting: uv run trading_floor.py"
    try {
        # Runs in the foreground -- trading_floor.py's own output goes straight
        # to this console, unmodified.
        uv run trading_floor.py
        $exitCode = $LASTEXITCODE
    } catch {
        Write-Log "Wrapper caught an exception launching trading_floor.py: $_"
        $exitCode = 1
    }

    if ($exitCode -eq 0) {
        # A clean exit is still unusual for a process meant to loop forever,
        # but it's not a crash -- don't burn a retry on it.
        Write-Log "trading_floor.py exited normally (code 0). Not restarting."
        break
    }

    $failureCount++
    Write-Log "trading_floor.py exited with code $exitCode (failure #$failureCount)."

    if ($failureCount -gt $MaxRestarts) {
        Write-Log "Already retried $MaxRestarts time(s) with no success. Giving up -- not restarting again."
        break
    }

    Write-Log "Restarting in $RestartDelaySeconds seconds... (retry $failureCount of $MaxRestarts)"
    Start-Sleep -Seconds $RestartDelaySeconds
}

Write-Log "Wrapper exiting."
