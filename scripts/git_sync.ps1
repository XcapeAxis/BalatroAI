param(
  [string]$Remote = "origin",
  [string]$BaseBranch = "",
  [switch]$PushCurrentBranch = $true,
  [switch]$PushTags = $true,
  [switch]$PruneRemote = $true,
  [switch]$DeleteMerged = $true,
  [switch]$DeleteGone = $true,
  [string]$DryRun = "true",
  [switch]$Force = $false,
  [switch]$ForceLocalCleanup = $false,
  [string[]]$ProtectBranches = @("main", "master", "dev", "develop", "release/*", "hotfix/*")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function ConvertTo-BoolFlag {
  param(
    [Parameter(Mandatory = $true)]
    [object]$Value,
    [bool]$Default = $false
  )

  if ($null -eq $Value) { return $Default }
  if ($Value -is [bool]) { return [bool]$Value }

  $s = [string]$Value
  if ([string]::IsNullOrWhiteSpace($s)) { return $Default }

  switch ($s.Trim().ToLowerInvariant()) {
    "true" { return $true }
    "1" { return $true }
    "yes" { return $true }
    "y" { return $true }
    "false" { return $false }
    "0" { return $false }
    "no" { return $false }
    "n" { return $false }
    default { return $Default }
  }
}

function Test-BranchExistsLocal([string]$BranchName) {
  & git show-ref --verify --quiet ("refs/heads/" + $BranchName) 2>$null
  return ($LASTEXITCODE -eq 0)
}

function Test-BranchExistsRemote([string]$RemoteName, [string]$BranchName) {
  & git show-ref --verify --quiet ("refs/remotes/" + $RemoteName + "/" + $BranchName) 2>$null
  return ($LASTEXITCODE -eq 0)
}

function Test-ProtectedBranch([string]$BranchName, [string[]]$Patterns) {
  foreach ($p in $Patterns) {
    if ($BranchName -like $p) {
      return $true
    }
  }
  return $false
}

function Invoke-GitExec {
  param(
    [string]$Name,
    [string]$Command,
    [bool]$ShouldExecute
  )

  $entry = [ordered]@{
    name = $Name
    command = $Command
    executed = $ShouldExecute
    success = $false
    output = @()
  }

  if (-not $ShouldExecute) {
    $entry.success = $true
    $entry.output = @("dry-run")
    return [pscustomobject]$entry
  }

  try {
    $out = & cmd /c $Command 2>&1
    $entry.output = @($out | ForEach-Object { [string]$_ })
    $entry.success = ($LASTEXITCODE -eq 0)
  } catch {
    $entry.output = @([string]$_.Exception.Message)
    $entry.success = $false
  }
  return [pscustomobject]$entry
}

$dryRunEnabled = ConvertTo-BoolFlag -Value $DryRun -Default $true

$repoRoot = (& git rev-parse --show-toplevel 2>$null)
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($repoRoot)) {
  throw "Not inside a git repository."
}
$repoRoot = $repoRoot.Trim()
Set-Location $repoRoot

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$artifactDir = Join-Path $repoRoot "docs/artifacts/git_sync"
if (-not (Test-Path $artifactDir)) {
  New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
}
$reportPath = Join-Path $artifactDir ("git_sync_" + $stamp + ".json")

$currentBranch = (& git rev-parse --abbrev-ref HEAD).Trim()
$remoteVerbose = @(& git remote -v 2>&1)
$remoteNames = @(& git remote 2>$null)
$hasRemote = $remoteNames -contains $Remote

$chosenBase = $BaseBranch
if ([string]::IsNullOrWhiteSpace($chosenBase)) {
  if (Test-BranchExistsLocal "main") {
    $chosenBase = "main"
  } elseif (Test-BranchExistsLocal "master") {
    $chosenBase = "master"
  } elseif (Test-BranchExistsRemote $Remote "main") {
    $chosenBase = "main"
  } else {
    $chosenBase = "master"
  }
}

$protectSet = @($ProtectBranches + $currentBranch) | Select-Object -Unique
$statusShort = @(& git status --short --branch 2>&1)
$branchVv = @(& git branch -vv 2>&1)

$planned = [ordered]@{
  fetch = $null
  ensure_base = $null
  pull_base = $null
  push_current_branch = $null
  push_tags = $null
  delete_gone = @()
  delete_merged = @()
}
$executed = New-Object System.Collections.Generic.List[object]

$localCleanupExecutionAllowed = $true
if (-not $hasRemote -and -not $dryRunEnabled -and -not $ForceLocalCleanup) {
  $localCleanupExecutionAllowed = $false
}

if ($PruneRemote) {
  $planned.fetch = [ordered]@{
    command = "git fetch $Remote --prune --tags"
    will_execute = ($hasRemote -and -not $dryRunEnabled)
  }
}

$baseExistsLocal = Test-BranchExistsLocal $chosenBase
$baseExistsRemote = Test-BranchExistsRemote $Remote $chosenBase

if (-not $baseExistsLocal -and $baseExistsRemote) {
  $planned.ensure_base = [ordered]@{
    command = "git branch --track $chosenBase $Remote/$chosenBase"
    will_execute = (-not $dryRunEnabled)
  }
}

$workingTreeClean = [string]::IsNullOrEmpty((& git status --porcelain))
$pullCanExecute = $false
$pullReason = ""
if ($hasRemote -and ($baseExistsLocal -or $baseExistsRemote)) {
  if ($dryRunEnabled) {
    $pullCanExecute = $false
  } elseif ($currentBranch -eq $chosenBase) {
    $pullCanExecute = $true
  } elseif ($workingTreeClean) {
    $pullCanExecute = $true
  } else {
    $pullCanExecute = $false
    $pullReason = "working tree not clean; skipping checkout/pull"
  }
}
$planned.pull_base = [ordered]@{
  command = "git pull --ff-only $Remote $chosenBase"
  will_execute = $pullCanExecute
  reason = $pullReason
}

$planned.push_current_branch = [ordered]@{
  command = "git push -u $Remote $currentBranch"
  will_execute = ($PushCurrentBranch -and $hasRemote -and -not $dryRunEnabled)
}
$planned.push_tags = [ordered]@{
  command = "git push $Remote --tags"
  will_execute = ($PushTags -and $hasRemote -and -not $dryRunEnabled)
}

$goneCandidates = New-Object System.Collections.Generic.List[string]
if ($DeleteGone) {
  $lines = @(& git for-each-ref refs/heads --format='%(refname:short)|%(upstream:short)|%(upstream:track)' 2>$null)
  foreach ($line in $lines) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    $parts = $line.Split("|")
    if ($parts.Count -lt 3) { continue }
    $bn = $parts[0].Trim()
    $track = $parts[2].Trim()
    if ($track -notmatch "\[gone\]") { continue }
    if (Test-ProtectedBranch $bn $protectSet) { continue }
    $goneCandidates.Add($bn)
  }
}
$planned.delete_gone = @($goneCandidates)

$mergedCandidates = New-Object System.Collections.Generic.List[string]
if ($DeleteMerged) {
  $mergeRef = if (Test-BranchExistsLocal $chosenBase) { $chosenBase } elseif (Test-BranchExistsRemote $Remote $chosenBase) { "$Remote/$chosenBase" } else { "" }
  if (-not [string]::IsNullOrWhiteSpace($mergeRef)) {
    $mergedLines = @(& git branch --merged $mergeRef 2>$null)
    foreach ($line in $mergedLines) {
      $bn = $line.Replace("*", "").Trim()
      if ([string]::IsNullOrWhiteSpace($bn)) { continue }
      if (Test-ProtectedBranch $bn $protectSet) { continue }
      $mergedCandidates.Add($bn)
    }
  }
}
$planned.delete_merged = @($mergedCandidates | Select-Object -Unique)

Write-Host ("repo_root: " + $repoRoot)
Write-Host ("remote: " + $Remote)
Write-Host ("base_branch: " + $chosenBase)
Write-Host ("current_branch: " + $currentBranch)
Write-Host ("dry_run: " + $dryRunEnabled)
if (-not $hasRemote) {
  Write-Host ("remote missing: " + $Remote + ". Configure remote first.")
}
Write-Host ("planned delete gone: " + (($planned.delete_gone -join ", ") -replace "^$", "<none>"))
Write-Host ("planned delete merged: " + (($planned.delete_merged -join ", ") -replace "^$", "<none>"))
if ($PushCurrentBranch) { Write-Host ("planned push branch: " + $currentBranch) }
if ($PushTags) { Write-Host "planned push tags: true" }

if ($PruneRemote -and $hasRemote) {
  $executed.Add((Invoke-GitExec -Name "fetch_prune" -Command ("git fetch " + $Remote + " --prune --tags") -ShouldExecute (-not $dryRunEnabled)))
}

if (-not $baseExistsLocal -and $baseExistsRemote) {
  $executed.Add((Invoke-GitExec -Name "ensure_base_branch" -Command ("git branch --track " + $chosenBase + " " + $Remote + "/" + $chosenBase) -ShouldExecute (-not $dryRunEnabled)))
}

if ($hasRemote) {
  if (-not $dryRunEnabled -and $pullCanExecute) {
    if ($currentBranch -ne $chosenBase) {
      $executed.Add((Invoke-GitExec -Name "checkout_base" -Command ("git checkout " + $chosenBase) -ShouldExecute $true))
    }
    $executed.Add((Invoke-GitExec -Name "pull_base_ff_only" -Command ("git pull --ff-only " + $Remote + " " + $chosenBase) -ShouldExecute $true))
    if ($currentBranch -ne $chosenBase) {
      $executed.Add((Invoke-GitExec -Name "checkout_restore_current" -Command ("git checkout " + $currentBranch) -ShouldExecute $true))
    }
  } elseif ($dryRunEnabled) {
    $executed.Add((Invoke-GitExec -Name "pull_base_ff_only" -Command ("git pull --ff-only " + $Remote + " " + $chosenBase) -ShouldExecute $false))
  } elseif (-not $pullCanExecute) {
    $executed.Add([pscustomobject]@{
      name = "pull_base_ff_only"
      command = "git pull --ff-only $Remote $chosenBase"
      executed = $false
      success = $false
      output = @("skipped: " + $pullReason)
    })
  }
}

if ($PushCurrentBranch -and $hasRemote) {
  $executed.Add((Invoke-GitExec -Name "push_current_branch" -Command ("git push -u " + $Remote + " " + $currentBranch) -ShouldExecute (-not $dryRunEnabled)))
}
if ($PushTags -and $hasRemote) {
  $executed.Add((Invoke-GitExec -Name "push_tags" -Command ("git push " + $Remote + " --tags") -ShouldExecute (-not $dryRunEnabled)))
}

$deleteCmd = if ($Force) { "-D" } else { "-d" }
foreach ($bn in $planned.delete_gone) {
  $canExec = ((-not $dryRunEnabled) -and $localCleanupExecutionAllowed)
  $executed.Add((Invoke-GitExec -Name ("delete_gone:" + $bn) -Command ("git branch " + $deleteCmd + " " + $bn) -ShouldExecute $canExec))
}
foreach ($bn in $planned.delete_merged) {
  if ($planned.delete_gone -contains $bn) { continue }
  $canExec = ((-not $dryRunEnabled) -and $localCleanupExecutionAllowed)
  $executed.Add((Invoke-GitExec -Name ("delete_merged:" + $bn) -Command ("git branch " + $deleteCmd + " " + $bn) -ShouldExecute $canExec))
}

$okCount = @($executed | Where-Object { $_.success }).Count
$failCount = @($executed | Where-Object { -not $_.success }).Count

$summary = [ordered]@{
  remote_exists = $hasRemote
  dry_run = $dryRunEnabled
  force_delete = [bool]$Force
  local_cleanup_execution_allowed = $localCleanupExecutionAllowed
  planned_delete_gone = @($planned.delete_gone).Count
  planned_delete_merged = @($planned.delete_merged).Count
  executed_actions = @($executed | Where-Object { $_.executed }).Count
  ok = $okCount
  failed = $failCount
}

$report = [ordered]@{
  timestamp = (Get-Date).ToString("o")
  repo_root = $repoRoot
  params = [ordered]@{
    remote = $Remote
    base_branch = $chosenBase
    push_current_branch = [bool]$PushCurrentBranch
    push_tags = [bool]$PushTags
    prune_remote = [bool]$PruneRemote
    delete_merged = [bool]$DeleteMerged
    delete_gone = [bool]$DeleteGone
    dry_run = $dryRunEnabled
    force = [bool]$Force
    force_local_cleanup = [bool]$ForceLocalCleanup
    protect_branches = $protectSet
  }
  pre_state = [ordered]@{
    status_short = $statusShort
    branch_vv = $branchVv
    remote_v = $remoteVerbose
    current_branch = $currentBranch
  }
  planned_actions = $planned
  executed_actions = $executed
  result_summary = $summary
}

$report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host ("report: " + $reportPath)
if ($failCount -gt 0) {
  Write-Host ("completed with failures: " + $failCount)
}

