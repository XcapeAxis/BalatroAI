param(
  [string]$BaseUrl = "http://127.0.0.1:12346",
  [string]$Seed = "AAAAAAA",
  [switch]$RequireMainBranch = $false,
  [string]$ArtifactTimestamp = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

function Invoke-CmdCapture([scriptblock]$Script) {
  $out = @(& $Script 2>&1)
  $code = $LASTEXITCODE
  return [pscustomobject]@{
    code = $code
    output = @($out | ForEach-Object { [string]$_ })
    text = (($out | ForEach-Object { [string]$_ }) -join "`n")
  }
}

function Invoke-GitCapture([string[]]$GitArgs) {
  $quoted = @($GitArgs | ForEach-Object {
    '"' + ([string]$_).Replace('"', '\"') + '"'
  })
  $command = "git " + ($quoted -join " ")
  $wrapped = $command + " 2>&1"
  $out = @(& cmd /c $wrapped)
  $code = $LASTEXITCODE
  return [pscustomobject]@{
    code = $code
    output = @($out | ForEach-Object { [string]$_ })
    text = (($out | ForEach-Object { [string]$_ }) -join "`n")
  }
}

function Invoke-PowerShellFile([string]$Label, [string]$FilePath, [string[]]$ScriptArgs) {
  $cmd = @("-ExecutionPolicy", "Bypass", "-File", $FilePath) + $ScriptArgs
  Write-Host ("[" + $Label + "] powershell " + ($cmd -join " "))
  $res = Invoke-CmdCapture -Script { & powershell @cmd }
  if ($res.output) { $res.output | ForEach-Object { Write-Host $_ } }
  return $res
}

function Write-JsonFile([string]$Path, $Object) {
  $dir = Split-Path -Parent $Path
  if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
  ($Object | ConvertTo-Json -Depth 24) | Out-File -LiteralPath $Path -Encoding UTF8
}

function Read-JsonOrNull([string]$Path) {
  if (-not (Test-Path $Path)) { return $null }
  try {
    return (Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json)
  } catch {
    return $null
  }
}

function Test-LocalBranchExists([string]$Branch) {
  $r = Invoke-GitCapture -GitArgs @("show-ref", "--verify", "--quiet", ("refs/heads/" + $Branch))
  return ($r.code -eq 0)
}

function Get-DetectedMainBranch() {
  $head = Invoke-GitCapture -GitArgs @("symbolic-ref", "refs/remotes/origin/HEAD")
  if ($head.code -eq 0 -and -not [string]::IsNullOrWhiteSpace($head.text)) {
    $raw = $head.text.Trim()
    if ($raw -match "^refs/remotes/origin/(.+)$") { return $Matches[1] }
  }
  if (Test-LocalBranchExists -Branch "main") { return "main" }
  if (Test-LocalBranchExists -Branch "master") { return "master" }
  return ""
}

function Get-HighestFunctionalRunP([string]$RunRegressionsPath) {
  if (-not (Test-Path $RunRegressionsPath)) { return 0 }
  $raw = Get-Content -LiteralPath $RunRegressionsPath -Raw
  $hits = [regex]::Matches($raw, "RunP(\d+)")
  $nums = New-Object System.Collections.Generic.List[int]
  foreach ($h in $hits) {
    [int]$v = 0
    if ([int]::TryParse($h.Groups[1].Value, [ref]$v)) {
      if ($v -ne 21) { $nums.Add($v) | Out-Null }
    }
  }
  if ($nums.Count -eq 0) { return 0 }
  return ($nums | Measure-Object -Maximum).Maximum
}

function Get-LatestGitSyncReportPath([string]$ProjectRootPath) {
  $dir = Join-Path $ProjectRootPath "docs/artifacts/git_sync"
  if (-not (Test-Path $dir)) { return "" }
  $latest = Get-ChildItem -Path $dir -Filter "git_sync_*.json" -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if (-not $latest) { return "" }
  return $latest.FullName
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

$stamp = if ([string]::IsNullOrWhiteSpace($ArtifactTimestamp)) { Get-Date -Format "yyyyMMdd-HHmmss" } else { $ArtifactTimestamp }
$artifactDir = Join-Path $ProjectRoot ("docs/artifacts/p21/" + $stamp)
if (-not (Test-Path $artifactDir)) { New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null }

$mainBranch = Get-DetectedMainBranch
$currentBranch = (Invoke-GitCapture -GitArgs @("rev-parse", "--abbrev-ref", "HEAD")).text.Trim()
if ([string]::IsNullOrWhiteSpace($mainBranch)) {
  $diag = [ordered]@{
    schema = "p21_report_v1"
    status = "FAIL"
    reason = "detect_main_branch_failed"
    current_branch = $currentBranch
    local_branches = @(Invoke-GitCapture -GitArgs @("branch", "--format=%(refname:short)")).output
    origin_refs = @(Invoke-GitCapture -GitArgs @("for-each-ref", "refs/remotes/origin", "--format=%(refname:short)")).output
  }
  Write-JsonFile -Path (Join-Path $artifactDir "report_p21.json") -Object $diag
  exit 21
}
if ($RequireMainBranch -and $currentBranch -ne $mainBranch) {
  throw ("[P21] RequireMainBranch enabled; switch first: git checkout " + $mainBranch)
}

$runRegressions = Join-Path $ProjectRoot "scripts/run_regressions.ps1"
$highest = Get-HighestFunctionalRunP -RunRegressionsPath $runRegressions
if ($highest -le 0) {
  throw "[P21] failed to detect highest RunP* from run_regressions.ps1"
}
$highestFlag = "-RunP$highest"

$baselineArgs = @($highestFlag)
if ($RequireMainBranch) { $baselineArgs += "-RequireMainBranch" }
$baselineResult = Invoke-PowerShellFile -Label "P21-baseline" -FilePath $runRegressions -ScriptArgs $baselineArgs
$baselineSummary = [ordered]@{
  schema = "p21_baseline_summary_v1"
  generated_at = (Get-Date).ToString("o")
  repo_root = $ProjectRoot
  branch_name = $currentBranch
  mainline_mode = $true
  detected_main_branch = $mainBranch
  highest_functional_gate = ("RunP" + $highest)
  command = ("powershell -ExecutionPolicy Bypass -File scripts/run_regressions.ps1 " + ($baselineArgs -join " "))
  exit_code = $baselineResult.code
  status = $(if ($baselineResult.code -eq 0) { "PASS" } else { "FAIL" })
}
Write-JsonFile -Path (Join-Path $artifactDir "baseline_summary.json") -Object $baselineSummary
@(
  "# P21 Baseline Summary",
  "",
  "- highest_functional_gate: RunP$highest",
  "- status: $($baselineSummary.status)",
  "- exit_code: $($baselineSummary.exit_code)",
  "- branch_name: $currentBranch",
  "- mainline_mode: true"
) | Out-File -LiteralPath (Join-Path $artifactDir "baseline_summary.md") -Encoding UTF8

$cleanupScript = Join-Path $ProjectRoot "scripts/git_branch_cleanup_mainline.ps1"
$cleanupDry = Invoke-PowerShellFile -Label "P21-cleanup-dryrun" -FilePath $cleanupScript -ScriptArgs @("-RepoRoot", $ProjectRoot, "-DryRun:$true", "-ForceDelete:$true", "-WriteReport:$true", "-ArtifactTimestamp", $stamp)
$cleanupReal = Invoke-PowerShellFile -Label "P21-cleanup-real" -FilePath $cleanupScript -ScriptArgs @("-RepoRoot", $ProjectRoot, "-DryRun:$false", "-ForceDelete:$true", "-WriteReport:$true", "-ArtifactTimestamp", $stamp)

$statusScript = Join-Path $ProjectRoot "scripts/git_mainline_status.ps1"
$statusJsonPath = Join-Path $artifactDir "git_mainline_status.json"
$statusRes = Invoke-PowerShellFile -Label "P21-mainline-status" -FilePath $statusScript -ScriptArgs @("-RepoRoot", $ProjectRoot, "-OutputPath", $statusJsonPath)
$mainlineStatus = Read-JsonOrNull -Path $statusJsonPath

$shortArgs = @("-RunFast")
if ($RequireMainBranch) { $shortArgs += "-RequireMainBranch" }
$shortResult = Invoke-PowerShellFile -Label "P21-short-gate" -FilePath $runRegressions -ScriptArgs $shortArgs

$gitSyncScript = Join-Path $ProjectRoot "scripts/git_sync.ps1"
$gitSyncRes = Invoke-PowerShellFile -Label "P21-gitsync-dryrun" -FilePath $gitSyncScript -ScriptArgs @("-DryRun:$true")
$latestGitSync = Get-LatestGitSyncReportPath -ProjectRootPath $ProjectRoot
$gitSyncReportDest = Join-Path $artifactDir "git_sync_dryrun_report.json"
if (-not [string]::IsNullOrWhiteSpace($latestGitSync) -and (Test-Path $latestGitSync)) {
  Copy-Item -LiteralPath $latestGitSync -Destination $gitSyncReportDest -Force
}

$beforePath = Join-Path $artifactDir "branch_cleanup_before.json"
$afterPath = Join-Path $artifactDir "branch_cleanup_after.json"
$summaryPath = Join-Path $artifactDir "branch_cleanup_summary.md"
$cleanupBefore = Read-JsonOrNull -Path $beforePath
$cleanupAfter = Read-JsonOrNull -Path $afterPath

$beforeBranches = @()
$deletedBranches = @()
$remainingBranches = @()
if ($cleanupBefore -and $cleanupBefore.local_branches_before) { $beforeBranches = @($cleanupBefore.local_branches_before) }
if ($cleanupAfter -and $cleanupAfter.local_branches_deleted) { $deletedBranches = @($cleanupAfter.local_branches_deleted) }
if ($cleanupAfter -and $cleanupAfter.local_branches_remaining) { $remainingBranches = @($cleanupAfter.local_branches_remaining) }
$onlyMainRemains = ($remainingBranches.Count -eq 1 -and $remainingBranches[0] -eq $mainBranch)

$gitSyncNetworkTolerated = $false
if ($gitSyncRes.code -ne 0) {
  if ($gitSyncRes.text -match "502" -or $gitSyncRes.text -match "timed out" -or $gitSyncRes.text -match "Unable to connect") {
    $gitSyncNetworkTolerated = $true
  }
}
$gitSyncPass = ($gitSyncRes.code -eq 0) -or $gitSyncNetworkTolerated

$workingTreeCleanNow = [bool]$(if ($mainlineStatus) { $mainlineStatus.working_tree_clean } else { $false })
$canCommitNow = [bool]$(if ($mainlineStatus) { $mainlineStatus.can_commit_now } else { $false })
$canCommitWhenCleanPass = $true
if ($workingTreeCleanNow) {
  $canCommitWhenCleanPass = $canCommitNow
}

$functionalPass = ($baselineResult.code -eq 0 -and $shortResult.code -eq 0)
$workflowPass = (
  $cleanupDry.code -eq 0 -and
  $cleanupReal.code -eq 0 -and
  $onlyMainRemains -and
  $mainlineStatus -ne $null -and
  [bool]$canCommitWhenCleanPass -and
  (Test-Path $beforePath) -and
  (Test-Path $afterPath) -and
  (Test-Path $summaryPath) -and
  $gitSyncPass
)

$gateFunctional = [ordered]@{
  schema = "p21_gate_functional_v1"
  generated_at = (Get-Date).ToString("o")
  branch_name = $currentBranch
  mainline_mode = $true
  status = $(if ($functionalPass) { "PASS" } else { "FAIL" })
  highest_gate = ("RunP" + $highest)
  baseline_exit_code = $baselineResult.code
  short_gate = "RunFast"
  short_gate_exit_code = $shortResult.code
  post_cleanup_regression_fail = [bool]($shortResult.code -ne 0)
}
Write-JsonFile -Path (Join-Path $artifactDir "gate_functional.json") -Object $gateFunctional

$gateGit = [ordered]@{
  schema = "p21_gate_git_workflow_v1"
  generated_at = (Get-Date).ToString("o")
  branch_name = $currentBranch
  mainline_mode = $true
  detected_main_branch = $mainBranch
  status = $(if ($workflowPass) { "PASS" } else { "FAIL" })
  checks = [ordered]@{
    local_only_main_branch = [bool]$onlyMainRemains
    can_commit_now = [bool]$canCommitNow
    working_tree_clean = [bool]$workingTreeCleanNow
    can_commit_when_clean_pass = [bool]$canCommitWhenCleanPass
    git_sync_dryrun_ok = [bool]$gitSyncPass
    git_sync_network_tolerated = [bool]$gitSyncNetworkTolerated
    cleanup_reports_present = [bool]((Test-Path $beforePath) -and (Test-Path $afterPath) -and (Test-Path $summaryPath))
  }
  branch_counts = [ordered]@{
    before = $beforeBranches.Count
    deleted = $deletedBranches.Count
    remaining = $remainingBranches.Count
  }
}
Write-JsonFile -Path (Join-Path $artifactDir "gate_git_workflow.json") -Object $gateGit

$reportP21 = [ordered]@{
  schema = "p21_report_v1"
  generated_at = (Get-Date).ToString("o")
  repo_root = $ProjectRoot
  mode = "MAINLINE_ONLY"
  detected_main_branch = $mainBranch
  branch_name = $currentBranch
  artifact_dir = $artifactDir
  gate_functional = $gateFunctional
  gate_git_workflow = $gateGit
  steps = [ordered]@{
    baseline_exit_code = $baselineResult.code
    cleanup_dryrun_exit_code = $cleanupDry.code
    cleanup_real_exit_code = $cleanupReal.code
    mainline_status_exit_code = $statusRes.code
    short_gate_exit_code = $shortResult.code
    git_sync_dryrun_exit_code = $gitSyncRes.code
  }
  post_cleanup_regression_fail = [bool]($shortResult.code -ne 0)
}
Write-JsonFile -Path (Join-Path $artifactDir "report_p21.json") -Object $reportP21

@(
  "# P21 Report",
  "",
  "- repo_root: $ProjectRoot",
  "- mode: MAINLINE_ONLY",
  "- detected_main_branch: $mainBranch",
  "- functional_gate: $($gateFunctional.status)",
  "- git_workflow_gate: $($gateGit.status)",
  "- baseline_gate: RunP$highest (exit=$($baselineResult.code))",
  "- short_gate: RunFast (exit=$($shortResult.code))",
  "- branch_counts_before_after: $($beforeBranches.Count) -> $($remainingBranches.Count)",
  "- git_sync_dryrun_exit: $($gitSyncRes.code)"
) | Out-File -LiteralPath (Join-Path $artifactDir "report_p21.md") -Encoding UTF8

$statusPanel = Join-Path $ProjectRoot "docs/COVERAGE_P21_STATUS.md"
@(
  "# P21 Status",
  "",
  "- status: $(if ($functionalPass -and $workflowPass) { 'PASS' } else { 'FAIL' })",
  "- updated_at_utc: $((Get-Date).ToUniversalTime().ToString('o'))",
  "- latest_artifact_dir: $artifactDir",
  "- detected_main_branch: $mainBranch",
  "- local_branches_before: $($beforeBranches.Count)",
  "- local_branches_deleted: $($deletedBranches.Count)",
  "- local_branches_after: $($remainingBranches.Count)",
  "- mode: MAINLINE_ONLY",
  "- functional_gate: $($gateFunctional.status)",
  "- git_workflow_gate: $($gateGit.status)"
) | Out-File -LiteralPath $statusPanel -Encoding UTF8

if (-not ($functionalPass -and $workflowPass)) {
  exit 1
}
