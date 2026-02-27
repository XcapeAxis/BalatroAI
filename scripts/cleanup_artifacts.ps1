param(
  [bool]$DryRun = $true,
  [int]$KeepRecent = 3,
  [int]$KeepRecentLogs = 3,
  [bool]$IncludeTrainerRuns = $true,
  [bool]$IncludeTrainerData = $true,
  [string]$OutDir = "docs/artifacts/cleanup"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

function Write-Json([string]$Path, $Obj) {
  $dir = Split-Path -Parent $Path
  if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
  ($Obj | ConvertTo-Json -Depth 12) | Out-File -LiteralPath $Path -Encoding UTF8
}

function Get-RelPath([string]$PathAbs) {
  $full = (Resolve-Path $PathAbs).Path
  $root = (Resolve-Path $ProjectRoot).Path
  if ($full.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
    return ($full.Substring($root.Length).TrimStart('\', '/')).Replace('\', '/')
  }
  return $full.Replace('\', '/')
}

function Get-PathSizeBytes([string]$PathAbs) {
  if (-not (Test-Path $PathAbs)) { return [int64]0 }
  $item = Get-Item -LiteralPath $PathAbs -Force
  if ($item.PSIsContainer) {
    $measure = Get-ChildItem -LiteralPath $PathAbs -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum
    $sum = $null
    if ($measure -and $measure.PSObject.Properties["Sum"]) {
      $sum = $measure.Sum
    }
    if ($null -eq $sum) { return [int64]0 }
    return [int64]$sum
  }
  return [int64]$item.Length
}

function Test-IsPassArtifactDir([string]$DirPath) {
  $gates = @("gate_perf.json", "gate_reliability.json", "gate_functional.json")
  foreach ($g in $gates) {
    $p = Join-Path $DirPath $g
    if (-not (Test-Path $p)) { continue }
    try {
      $obj = Get-Content -LiteralPath $p -Raw | ConvertFrom-Json
      $status = [string]($obj.status)
      if ($status.ToUpperInvariant() -eq "PASS") { return $true }
    } catch {}
  }
  return $false
}

function Get-TimestampSuffix([string]$Name) {
  if ($Name -match "_(\d{8}-\d{6})") {
    return $Matches[1]
  }
  return ""
}

function Collect-StringValues($Value, [System.Collections.Generic.List[string]]$Out) {
  if ($null -eq $Value) { return }
  if ($Value -is [string]) {
    $Out.Add([string]$Value)
    return
  }
  if ($Value -is [System.Collections.IDictionary]) {
    foreach ($k in $Value.Keys) {
      Collect-StringValues -Value $Value[$k] -Out $Out
    }
    return
  }
  if ($Value -is [System.Collections.IEnumerable] -and -not ($Value -is [string])) {
    foreach ($v in $Value) {
      Collect-StringValues -Value $v -Out $Out
    }
  }
}

$tracked = New-Object System.Collections.Generic.HashSet[string]([System.StringComparer]::OrdinalIgnoreCase)
try {
  $files = & git ls-files
  foreach ($f in $files) {
    $tracked.Add(([string]$f).Replace('\', '/')) | Out-Null
  }
} catch {}

$candidates = New-Object System.Collections.Generic.List[object]

function Add-Candidate([string]$PathAbs, [string]$Reason) {
  if (-not (Test-Path $PathAbs)) { return }
  $rel = Get-RelPath -PathAbs $PathAbs
  $item = Get-Item -LiteralPath $PathAbs -Force
  $kind = if ($item.PSIsContainer) { "dir" } else { "file" }
  $size = Get-PathSizeBytes -PathAbs $PathAbs
  $isTracked = $tracked.Contains($rel)
  $candidates.Add([ordered]@{
    path = $rel
    kind = $kind
    size_bytes = $size
    tracked = $isTracked
    reason = $Reason
  }) | Out-Null
}

# 1) docs/artifacts cleanup
$artRoot = Join-Path $ProjectRoot "docs/artifacts"
if (Test-Path $artRoot) {
  $phaseDirs = @(
    Get-ChildItem -LiteralPath $artRoot -Directory -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -notin @("cleanup") }
  )
  foreach ($phase in $phaseDirs) {
    $subDirs = @(
      Get-ChildItem -LiteralPath $phase.FullName -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -ne "cleanup" } |
        Sort-Object LastWriteTime -Descending
    )

    if ($subDirs.Count -gt 0) {
      $keep = New-Object System.Collections.Generic.HashSet[string]([System.StringComparer]::OrdinalIgnoreCase)
      $subDirs | Select-Object -First $KeepRecent | ForEach-Object { $keep.Add($_.FullName) | Out-Null }
      $latestPass = $null
      foreach ($d in $subDirs) {
        if (Test-IsPassArtifactDir -DirPath $d.FullName) { $latestPass = $d.FullName; break }
      }
      if ($latestPass) { $keep.Add($latestPass) | Out-Null }

      foreach ($d in $subDirs) {
        if (-not $keep.Contains($d.FullName)) {
          Add-Candidate -PathAbs $d.FullName -Reason ("artifacts phase retention (keep recent {0} + latest pass)" -f $KeepRecent)
        }
      }
    }

    # Handle flat timestamped files (legacy p3/p4 pattern)
    $phaseFiles = @(
      Get-ChildItem -LiteralPath $phase.FullName -File -ErrorAction SilentlyContinue
    )
    if ($phaseFiles.Count -gt 0) {
      $groups = @{}
      foreach ($f in $phaseFiles) {
        $ts = Get-TimestampSuffix -Name $f.Name
        if (-not $ts) { continue }
        if (-not $groups.ContainsKey($ts)) { $groups[$ts] = New-Object System.Collections.Generic.List[object] }
        $groups[$ts].Add($f) | Out-Null
      }
      $orderedTs = @($groups.Keys | Sort-Object -Descending)
      if ($orderedTs.Count -gt $KeepRecent) {
        $dropTs = @($orderedTs | Select-Object -Skip $KeepRecent)
        foreach ($ts in $dropTs) {
          foreach ($f in $groups[$ts]) {
            Add-Candidate -PathAbs $f.FullName -Reason ("legacy timestamped artifact retention (keep recent {0})" -f $KeepRecent)
          }
        }
      }
    }
  }
}

# 2) logs cleanup
$logsRoot = Join-Path $ProjectRoot "logs"
if (Test-Path $logsRoot) {
  $logDirs = @(
    Get-ChildItem -LiteralPath $logsRoot -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
  )
  $dropLogDirs = @($logDirs | Select-Object -Skip $KeepRecentLogs)
  foreach ($d in $dropLogDirs) {
    Add-Candidate -PathAbs $d.FullName -Reason ("logs retention (keep recent {0})" -f $KeepRecentLogs)
  }
}

# 3) trainer_runs cleanup
if ($IncludeTrainerRuns) {
  $runsRoot = Join-Path $ProjectRoot "trainer_runs"
  if (Test-Path $runsRoot) {
    $preserveModelPaths = New-Object System.Collections.Generic.HashSet[string]([System.StringComparer]::OrdinalIgnoreCase)
    $refRoots = @(
      (Join-Path $ProjectRoot "docs/artifacts/p18"),
      (Join-Path $ProjectRoot "docs/artifacts/p19"),
      (Join-Path $ProjectRoot "docs/artifacts/p20")
    )
    foreach ($r in $refRoots) {
      if (-not (Test-Path $r)) { continue }
      $latest = @(
        Get-ChildItem -LiteralPath $r -Directory -ErrorAction SilentlyContinue |
          Sort-Object LastWriteTime -Descending |
          Select-Object -First 5
      )
      foreach ($d in $latest) {
        $jsons = @(
          Get-ChildItem -LiteralPath $d.FullName -Recurse -File -Include *.json -ErrorAction SilentlyContinue
        )
        foreach ($j in $jsons) {
          try {
            $payload = Get-Content -LiteralPath $j.FullName -Raw | ConvertFrom-Json
            $vals = New-Object System.Collections.Generic.List[string]
            Collect-StringValues -Value $payload -Out $vals
            foreach ($s in $vals) {
              if ([string]::IsNullOrWhiteSpace($s)) { continue }
              if (-not $s.ToLowerInvariant().EndsWith(".pt")) { continue }
              if (Test-Path $s) {
                $preserveModelPaths.Add((Resolve-Path $s).Path) | Out-Null
              }
            }
          } catch {}
        }
      }
    }

    $runDirs = @(
      Get-ChildItem -LiteralPath $runsRoot -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
    )
    $keepRunDirs = New-Object System.Collections.Generic.HashSet[string]([System.StringComparer]::OrdinalIgnoreCase)
    $runDirs | Select-Object -First $KeepRecent | ForEach-Object { $keepRunDirs.Add($_.FullName) | Out-Null }
    foreach ($pt in $preserveModelPaths) {
      $parent = Split-Path -Parent $pt
      if ($parent) { $keepRunDirs.Add($parent) | Out-Null }
    }
    foreach ($d in $runDirs) {
      if (-not $keepRunDirs.Contains($d.FullName)) {
        Add-Candidate -PathAbs $d.FullName -Reason ("trainer_runs retention (keep recent {0} + referenced models)" -f $KeepRecent)
      }
    }
  }
}

# 4) trainer_data cleanup
if ($IncludeTrainerData) {
  $dataRoot = Join-Path $ProjectRoot "trainer_data"
  if (Test-Path $dataRoot) {
    $keepNames = New-Object System.Collections.Generic.HashSet[string]([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($n in @("p20_distill_smoke.jsonl", "p19_dagger_v4.jsonl", "p18_dagger_v3.jsonl")) { $keepNames.Add($n) | Out-Null }
    $allFiles = @(
      Get-ChildItem -LiteralPath $dataRoot -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
    )
    $allFiles | Select-Object -First $KeepRecent | ForEach-Object { $keepNames.Add($_.Name) | Out-Null }
    foreach ($f in $allFiles) {
      if (-not $keepNames.Contains($f.Name)) {
        Add-Candidate -PathAbs $f.FullName -Reason ("trainer_data retention (keep recent {0} + active datasets)" -f $KeepRecent)
      }
    }
  }
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$outRoot = Join-Path $ProjectRoot $OutDir
if (-not (Test-Path $outRoot)) { New-Item -ItemType Directory -Path $outRoot -Force | Out-Null }
$candidatePath = Join-Path $outRoot ("cleanup_candidates_{0}.json" -f $stamp)

$totalBytes = [int64]0
$trackedCount = 0
foreach ($c in $candidates) {
  $totalBytes += [int64]$c.size_bytes
  if ([bool]$c.tracked) { $trackedCount++ }
}

$candidatePayload = [ordered]@{
  schema = "cleanup_candidates_v1"
  generated_at = (Get-Date).ToString("o")
  dry_run = $DryRun
  keep_recent = $KeepRecent
  keep_recent_logs = $KeepRecentLogs
  candidate_count = $candidates.Count
  tracked_candidate_count = $trackedCount
  reclaimable_bytes_estimate = $totalBytes
  candidates = @($candidates.ToArray() | Sort-Object size_bytes -Descending)
}
Write-Json -Path $candidatePath -Obj $candidatePayload
Write-Host ("[cleanup] candidates: {0}" -f $candidatePath)
Write-Host ("[cleanup] count={0} tracked={1} bytes~{2}" -f $candidates.Count, $trackedCount, $totalBytes)

if ($DryRun) {
  Write-Host "[cleanup] dry-run only; no files deleted."
  exit 0
}

$deleted = New-Object System.Collections.Generic.List[object]
$failed = New-Object System.Collections.Generic.List[object]
$freed = [int64]0
foreach ($c in ($candidates | Sort-Object size_bytes -Descending)) {
  $abs = Join-Path $ProjectRoot ([string]$c.path).Replace("/", "\")
  try {
    if (Test-Path $abs) {
      if ([string]$c.kind -eq "dir") {
        Remove-Item -LiteralPath $abs -Recurse -Force
      } else {
        Remove-Item -LiteralPath $abs -Force
      }
      $freed += [int64]$c.size_bytes
      $deleted.Add($c) | Out-Null
    }
  } catch {
    $failed.Add([ordered]@{
      path = $c.path
      reason = $_.Exception.Message
    }) | Out-Null
  }
}

$reportPath = Join-Path $outRoot ("cleanup_report_{0}.json" -f $stamp)
$reportPayload = [ordered]@{
  schema = "cleanup_report_v1"
  generated_at = (Get-Date).ToString("o")
  dry_run = $false
  candidate_manifest = $candidatePath
  deleted_count = $deleted.Count
  failed_count = $failed.Count
  freed_bytes = $freed
  deleted = @($deleted.ToArray())
  failed = @($failed.ToArray())
}
Write-Json -Path $reportPath -Obj $reportPayload
Write-Host ("[cleanup] report: {0}" -f $reportPath)
if ($failed.Count -gt 0) {
  Write-Host ("[cleanup] completed with failures: {0}" -f $failed.Count)
  exit 1
}

exit 0
