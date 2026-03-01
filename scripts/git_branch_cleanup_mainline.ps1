param(
  [string]$DryRun = "true",
  [string]$RepoRoot = "",
  [string[]]$KeepBranches = @(),
  [string]$ForceDelete = "true",
  [string]$WriteReport = "true",
  [string]$ArtifactTimestamp = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

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

function Invoke-GitCapture([string[]]$GitArgs) {
  $quoted = @($GitArgs | ForEach-Object {
    '"' + ([string]$_).Replace('"', '\"') + '"'
  })
  $command = "git " + ($quoted -join " ")
  $wrapped = $command + " 2>&1"
  $output = @(& cmd /c $wrapped)
  $code = $LASTEXITCODE
  return [pscustomobject]@{
    code = $code
    output = @($output | ForEach-Object { [string]$_ })
    text = (($output | ForEach-Object { [string]$_ }) -join "`n")
  }
}

function Write-JsonFile([string]$Path, $Object) {
  $parent = Split-Path -Parent $Path
  if ($parent -and -not (Test-Path $parent)) {
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
  }
  ($Object | ConvertTo-Json -Depth 24) | Out-File -LiteralPath $Path -Encoding UTF8
}

function Get-RepoRoot([string]$Provided) {
  if (-not [string]::IsNullOrWhiteSpace($Provided)) {
    if (-not (Test-Path $Provided)) {
      throw "repo root not found: $Provided"
    }
    return (Resolve-Path $Provided).Path
  }
  $r = Invoke-GitCapture -GitArgs @("rev-parse", "--show-toplevel")
  if ($r.code -ne 0 -or [string]::IsNullOrWhiteSpace($r.text)) {
    throw "not inside a git repository"
  }
  return $r.text.Trim()
}

function Test-LocalBranchExists([string]$Branch) {
  $r = Invoke-GitCapture -GitArgs @("show-ref", "--verify", "--quiet", ("refs/heads/" + $Branch))
  return ($r.code -eq 0)
}

function Get-DetectedMainBranch() {
  $originHead = Invoke-GitCapture -GitArgs @("symbolic-ref", "refs/remotes/origin/HEAD")
  if ($originHead.code -eq 0 -and -not [string]::IsNullOrWhiteSpace($originHead.text)) {
    $raw = $originHead.text.Trim()
    if ($raw -match "^refs/remotes/origin/(.+)$") {
      return $Matches[1]
    }
  }
  if (Test-LocalBranchExists -Branch "main") { return "main" }
  if (Test-LocalBranchExists -Branch "master") { return "master" }
  return $null
}

function Get-LocalBranches() {
  $r = Invoke-GitCapture -GitArgs @("for-each-ref", "refs/heads", "--format=%(refname:short)")
  if ($r.code -ne 0) { throw "failed to list local branches: $($r.text)" }
  return @($r.output | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | ForEach-Object { $_.Trim() })
}

function Get-BranchHeadSha([string]$Branch) {
  $r = Invoke-GitCapture -GitArgs @("rev-parse", $Branch)
  if ($r.code -ne 0) { return "" }
  return $r.text.Trim()
}

function Get-BranchUpstream([string]$Branch) {
  $r = Invoke-GitCapture -GitArgs @("rev-parse", "--abbrev-ref", ($Branch + "@{upstream}"))
  if ($r.code -ne 0) { return "" }
  return $r.text.Trim()
}

function Get-MergedTo([string]$TargetBranch) {
  $r = Invoke-GitCapture -GitArgs @("branch", "--merged", $TargetBranch, "--format=%(refname:short)")
  if ($r.code -ne 0) { return @() }
  return @($r.output | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | ForEach-Object { $_.Trim() })
}

function Build-BranchDetails([string[]]$Branches, [hashtable]$MergedMap) {
  $rows = @()
  foreach ($b in $Branches) {
    $rows += [ordered]@{
      branch = $b
      sha = (Get-BranchHeadSha -Branch $b)
      upstream = (Get-BranchUpstream -Branch $b)
      merged_to_main = [bool]$MergedMap.ContainsKey($b)
    }
  }
  return $rows
}

$resolvedRoot = Get-RepoRoot -Provided $RepoRoot
Set-Location $resolvedRoot

$dryRunEnabled = ConvertTo-BoolFlag -Value $DryRun -Default $true
$forceDeleteEnabled = ConvertTo-BoolFlag -Value $ForceDelete -Default $true
$writeReportEnabled = ConvertTo-BoolFlag -Value $WriteReport -Default $true

$mainBranch = Get-DetectedMainBranch
if (-not $mainBranch) {
  $diag = [ordered]@{
    schema = "p21_branch_cleanup_v1"
    status = "failed_main_branch_detection"
    exit_code = 21
    generated_at = (Get-Date).ToString("o")
    repo_root = $resolvedRoot
    local_branches = (Get-LocalBranches)
    remote_refs = (Invoke-GitCapture -GitArgs @("for-each-ref", "refs/remotes/origin", "--format=%(refname:short)")).output
  }
  Write-Host "[p21] failed to detect main branch"
  Write-Host ("local branches: " + (($diag.local_branches -join ", ") -replace "^$", "<none>"))
  Write-Host ("origin refs: " + (($diag.remote_refs -join ", ") -replace "^$", "<none>"))
  if ($writeReportEnabled) {
    $stamp = if ([string]::IsNullOrWhiteSpace($ArtifactTimestamp)) { Get-Date -Format "yyyyMMdd-HHmmss" } else { $ArtifactTimestamp }
    $artifactDir = Join-Path $resolvedRoot ("docs/artifacts/p21/" + $stamp)
    New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
    Write-JsonFile -Path (Join-Path $artifactDir "branch_cleanup_before.json") -Object $diag
    Write-JsonFile -Path (Join-Path $artifactDir "branch_cleanup_after.json") -Object $diag
  }
  exit 21
}

$stampUsed = if ([string]::IsNullOrWhiteSpace($ArtifactTimestamp)) { Get-Date -Format "yyyyMMdd-HHmmss" } else { $ArtifactTimestamp }
$artifactDir = Join-Path $resolvedRoot ("docs/artifacts/p21/" + $stampUsed)
if ($writeReportEnabled -and -not (Test-Path $artifactDir)) {
  New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
}

$keepSet = New-Object System.Collections.Generic.HashSet[string]([System.StringComparer]::OrdinalIgnoreCase)
[void]$keepSet.Add($mainBranch)
foreach ($k in $KeepBranches) {
  if (-not [string]::IsNullOrWhiteSpace($k)) {
    [void]$keepSet.Add($k.Trim())
  }
}

$currentBranch = (Invoke-GitCapture -GitArgs @("rev-parse", "--abbrev-ref", "HEAD")).text.Trim()
if ($currentBranch -ne $mainBranch) {
  $checkout = Invoke-GitCapture -GitArgs @("checkout", $mainBranch)
  if ($checkout.code -ne 0) {
    throw "failed to checkout main branch '$mainBranch': $($checkout.text)"
  }
  $currentBranch = $mainBranch
}

$branchesBefore = Get-LocalBranches
$mergedList = Get-MergedTo -TargetBranch $mainBranch
$mergedMap = @{}
foreach ($m in $mergedList) { $mergedMap[$m] = $true }
$detailsBefore = Build-BranchDetails -Branches $branchesBefore -MergedMap $mergedMap
$headCommitsBefore = @($detailsBefore | ForEach-Object {
  [ordered]@{
    branch = [string]$_.branch
    sha = [string]$_.sha
  }
})

$deleteCandidates = @($branchesBefore | Where-Object { -not $keepSet.Contains($_) })
$deleted = New-Object System.Collections.Generic.List[string]
$deleteErrors = New-Object System.Collections.Generic.List[object]

foreach ($branchName in $deleteCandidates) {
  if ($dryRunEnabled) {
    [void]$deleted.Add($branchName)
    continue
  }
  $flag = if ($forceDeleteEnabled) { "-D" } else { "-d" }
  $del = Invoke-GitCapture -GitArgs @("branch", $flag, $branchName)
  if ($del.code -eq 0) {
    [void]$deleted.Add($branchName)
  } else {
    $deleteErrors.Add([ordered]@{
      branch = $branchName
      command = ("git branch " + $flag + " " + $branchName)
      message = $del.text
      exit_code = $del.code
    }) | Out-Null
  }
}

$branchesAfter = Get-LocalBranches
$remainingUnexpected = @($branchesAfter | Where-Object { -not $keepSet.Contains($_) })
$status = "PASS"
$exitCode = 0
if ($deleteErrors.Count -gt 0) {
  $status = "FAIL"
  $exitCode = 22
}
if ((-not $dryRunEnabled) -and $remainingUnexpected.Count -gt 0) {
  $status = "FAIL"
  if ($exitCode -eq 0) { $exitCode = 22 }
}

$beforePayload = [ordered]@{
  schema = "p21_branch_cleanup_v1"
  status = "before"
  generated_at = (Get-Date).ToString("o")
  repo_root = $resolvedRoot
  detected_main_branch = $mainBranch
  dry_run = [bool]$dryRunEnabled
  force_delete = [bool]$forceDeleteEnabled
  keep_branches = @($keepSet | Sort-Object)
  current_branch = $currentBranch
  local_branches_before = $branchesBefore
  branch_head_commits = $headCommitsBefore
  branch_details = $detailsBefore
  local_branches_deleted = @($deleted.ToArray())
  local_branches_remaining = $branchesAfter
  local_branches_delete_candidates = $deleteCandidates
  delete_errors = @($deleteErrors.ToArray())
}

$afterPayload = [ordered]@{
  schema = "p21_branch_cleanup_v1"
  status = $status
  generated_at = (Get-Date).ToString("o")
  repo_root = $resolvedRoot
  detected_main_branch = $mainBranch
  dry_run = [bool]$dryRunEnabled
  force_delete = [bool]$forceDeleteEnabled
  keep_branches = @($keepSet | Sort-Object)
  local_branches_before = $branchesBefore
  local_branches_deleted = @($deleted.ToArray())
  local_branches_remaining = $branchesAfter
  branch_head_commits = $headCommitsBefore
  delete_errors = @($deleteErrors.ToArray())
  validation = [ordered]@{
    final_only_keep_branches = [bool]($remainingUnexpected.Count -eq 0)
    unexpected_remaining = $remainingUnexpected
  }
}

if ($writeReportEnabled) {
  $beforePath = Join-Path $artifactDir "branch_cleanup_before.json"
  $afterPath = Join-Path $artifactDir "branch_cleanup_after.json"
  $summaryPath = Join-Path $artifactDir "branch_cleanup_summary.md"
  Write-JsonFile -Path $beforePath -Object $beforePayload
  Write-JsonFile -Path $afterPath -Object $afterPayload

  $summaryLines = @(
    "# P21 Branch Cleanup Summary",
    "",
    "- repo_root: $resolvedRoot",
    "- detected_main_branch: $mainBranch",
    "- dry_run: $dryRunEnabled",
    "- force_delete: $forceDeleteEnabled",
    "- branches_before_count: $($branchesBefore.Count)",
    "- delete_candidates_count: $($deleteCandidates.Count)",
    "- deleted_count: $($deleted.Count)",
    "- branches_after_count: $($branchesAfter.Count)",
    "- status: $status",
    "- delete_error_count: $($deleteErrors.Count)"
  )
  if ($deleteCandidates.Count -gt 0) {
    $summaryLines += ""
    $summaryLines += "## Delete Candidates"
    foreach ($b in $deleteCandidates) {
      $sha = ($headCommitsBefore | Where-Object { $_.branch -eq $b } | Select-Object -First 1).sha
      $summaryLines += ("- " + $b + " (" + $sha + ")")
    }
  }
  if ($deleteErrors.Count -gt 0) {
    $summaryLines += ""
    $summaryLines += "## Delete Errors"
    foreach ($err in $deleteErrors) {
      $summaryLines += ("- " + $err.branch + ": " + $err.message)
    }
  }
  $summaryLines | Out-File -LiteralPath $summaryPath -Encoding UTF8
}

Write-Host ("repo_root: " + $resolvedRoot)
Write-Host ("detected_main_branch: " + $mainBranch)
Write-Host ("dry_run: " + $dryRunEnabled)
Write-Host ("branches_before: " + $branchesBefore.Count)
Write-Host ("delete_candidates: " + $deleteCandidates.Count)
Write-Host ("deleted: " + $deleted.Count)
Write-Host ("branches_after: " + $branchesAfter.Count)
if ($deleteErrors.Count -gt 0) {
  Write-Host ("delete_errors: " + $deleteErrors.Count)
}

if ($exitCode -ne 0) {
  exit $exitCode
}
