param(
  [string]$RepoRoot = "",
  [string]$OutputPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

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

function Resolve-RepoRoot([string]$Provided) {
  if (-not [string]::IsNullOrWhiteSpace($Provided)) {
    if (-not (Test-Path $Provided)) { throw "repo root not found: $Provided" }
    return (Resolve-Path $Provided).Path
  }
  $r = Invoke-GitCapture -GitArgs @("rev-parse", "--show-toplevel")
  if ($r.code -ne 0) { throw "not inside a git repository" }
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
  return ""
}

$resolvedRoot = Resolve-RepoRoot -Provided $RepoRoot
Set-Location $resolvedRoot

$currentBranch = (Invoke-GitCapture -GitArgs @("rev-parse", "--abbrev-ref", "HEAD")).text.Trim()
$mainBranch = Get-DetectedMainBranch
$statusLines = (Invoke-GitCapture -GitArgs @("status", "--porcelain")).output
$statusShort = (Invoke-GitCapture -GitArgs @("status", "--short", "--branch")).output
$workingTreeClean = (@($statusLines).Count -eq 0)
$untrackedCount = @($statusLines | Where-Object { $_ -like "?? *" }).Count

$ahead = 0
$behind = 0
$remoteReachable = $false
$remoteError = ""
$compareRef = ""

if (-not [string]::IsNullOrWhiteSpace($mainBranch)) {
  $compareRef = "origin/$mainBranch"
  $remoteCheck = Invoke-GitCapture -GitArgs @("show-ref", "--verify", "--quiet", ("refs/remotes/" + $compareRef))
  if ($remoteCheck.code -eq 0) {
    $cmp = Invoke-GitCapture -GitArgs @("rev-list", "--left-right", "--count", ($compareRef + "..." + $mainBranch))
    if ($cmp.code -eq 0 -and -not [string]::IsNullOrWhiteSpace($cmp.text)) {
      $parts = $cmp.text.Trim().Split(@(" ", "`t"), [System.StringSplitOptions]::RemoveEmptyEntries)
      if ($parts.Count -ge 2) {
        $behind = [int]$parts[0]
        $ahead = [int]$parts[1]
        $remoteReachable = $true
      }
    } else {
      $remoteError = $cmp.text
    }
  } else {
    $remoteError = "remote ref not found: $compareRef"
  }
}

$branchPolicyViolation = ($mainBranch -ne "" -and $currentBranch -ne $mainBranch)
$canCommitNow = (-not $branchPolicyViolation) -and $workingTreeClean

$nextStep = ""
if ([string]::IsNullOrWhiteSpace($mainBranch)) {
  $nextStep = "detect_main_branch_failed"
} elseif ($branchPolicyViolation) {
  $nextStep = ("switch_to_main_branch: git checkout " + $mainBranch)
} elseif (-not $workingTreeClean) {
  $nextStep = "working_tree_dirty_commit_or_stash"
} elseif ($remoteReachable -and $behind -gt 0) {
  $nextStep = ("sync_first: git pull --ff-only origin " + $mainBranch)
} else {
  $nextStep = "ready_to_commit"
}

$payload = [ordered]@{
  schema = "p21_git_mainline_status_v1"
  generated_at = (Get-Date).ToString("o")
  repo_root = $resolvedRoot
  current_branch = $currentBranch
  detected_main_branch = $mainBranch
  working_tree_clean = [bool]$workingTreeClean
  untracked_count = [int]$untrackedCount
  ahead = [int]$ahead
  behind = [int]$behind
  compare_ref = $compareRef
  remote_reachable = [bool]$remoteReachable
  remote_error = $remoteError
  branch_policy_violation = [bool]$branchPolicyViolation
  can_commit_now = [bool]$canCommitNow
  suggested_next_step = $nextStep
  status_short = $statusShort
}

if (-not [string]::IsNullOrWhiteSpace($OutputPath)) {
  Write-JsonFile -Path $OutputPath -Object $payload
}

Write-Host ("repo_root: " + $resolvedRoot)
Write-Host ("current_branch: " + $currentBranch)
Write-Host ("detected_main_branch: " + $mainBranch)
Write-Host ("working_tree_clean: " + $workingTreeClean)
Write-Host ("ahead: " + $ahead)
Write-Host ("behind: " + $behind)
Write-Host ("untracked_count: " + $untrackedCount)
Write-Host ("can_commit_now: " + $canCommitNow)
Write-Host ("suggested_next_step: " + $nextStep)

if (-not [string]::IsNullOrWhiteSpace($OutputPath)) {
  Write-Host ("json_report: " + $OutputPath)
}
