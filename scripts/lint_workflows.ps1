param(
  [string]$WorkflowDir = ".github/workflows",
  [string]$OutJson = "docs/artifacts/status/workflow_lint_report.json",
  [string]$OutMd = "docs/artifacts/status/workflow_lint_report.md"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

function Ensure-Dir([string]$Path) {
  $dir = Split-Path -Parent $Path
  if ($dir -and -not (Test-Path $dir)) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
  }
}

function Test-Pattern([string]$Text, [string]$Pattern) {
  return [regex]::IsMatch($Text, $Pattern, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
}

function Analyze-Workflow([string]$FilePath) {
  $raw = Get-Content -LiteralPath $FilePath -Raw
  $parseOk = $false
  try {
    $null = ConvertFrom-Yaml -Yaml $raw
    $parseOk = $true
  } catch {
    $py = "python"
    $probe = @'
import pathlib
import sys
import yaml

path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
yaml.safe_load(text)
print("ok")
'@
    $prevEA = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $probeOut = @($probe | & $py - $FilePath 2>&1)
    $nativeCode = $LASTEXITCODE
    $ErrorActionPreference = $prevEA
    if ($nativeCode -eq 0) {
      $parseOk = $true
    }
  }

  $hasName = Test-Pattern -Text $raw -Pattern "(?m)^\s*name\s*:"
  $hasOn = Test-Pattern -Text $raw -Pattern "(?m)^\s*on\s*:"
  $hasJobs = Test-Pattern -Text $raw -Pattern "(?m)^\s*jobs\s*:"

  $hasCheckout = Test-Pattern -Text $raw -Pattern "actions/checkout@"
  $hasSetupPython = Test-Pattern -Text $raw -Pattern "actions/setup-python@"
  $hasScriptExec = Test-Pattern -Text $raw -Pattern "(scripts/run_p26\.ps1|trainer\.experiments\.status_publish|scripts/update_readme_badges\.ps1|scripts/build_dashboard\.ps1|run_regressions\.ps1)"

  $issues = New-Object System.Collections.Generic.List[string]
  if (-not $parseOk) { $issues.Add("yaml_parse_failed") | Out-Null }
  if (-not $hasName) { $issues.Add("missing_name") | Out-Null }
  if (-not $hasOn) { $issues.Add("missing_on") | Out-Null }
  if (-not $hasJobs) { $issues.Add("missing_jobs") | Out-Null }
  if (-not $hasCheckout) { $issues.Add("missing_checkout_step") | Out-Null }
  if (-not $hasSetupPython) { $issues.Add("missing_setup_python_step") | Out-Null }
  if (-not $hasScriptExec) { $issues.Add("missing_script_execution_step") | Out-Null }

  return [ordered]@{
    file = $FilePath
    parse_ok = $parseOk
    has_name = $hasName
    has_on = $hasOn
    has_jobs = $hasJobs
    has_checkout = $hasCheckout
    has_setup_python = $hasSetupPython
    has_script_execution = $hasScriptExec
    issue_count = $issues.Count
    issues = @($issues)
    pass = ($issues.Count -eq 0)
  }
}

$workflowRoot = Join-Path $ProjectRoot $WorkflowDir
if (-not (Test-Path $workflowRoot)) {
  throw "workflow directory not found: $workflowRoot"
}

$requiredFiles = @(
  (Join-Path $workflowRoot "ci-smoke.yml"),
  (Join-Path $workflowRoot "nightly-orchestrator.yml")
)

$missingRequired = @($requiredFiles | Where-Object { -not (Test-Path $_) })
$workflowFiles = Get-ChildItem -Path $workflowRoot -Filter "*.yml" -File -ErrorAction SilentlyContinue | Sort-Object Name
$reports = @()
foreach ($wf in $workflowFiles) {
  $reports += ,(Analyze-Workflow -FilePath $wf.FullName)
}

$failCount = @($reports | Where-Object { -not $_.pass }).Count
$issueCount = 0
foreach ($r in $reports) {
  $issueCount += [int]$r.issue_count
}

if ($missingRequired.Count -gt 0) {
  $failCount += $missingRequired.Count
  $issueCount += $missingRequired.Count
}

$overallPass = ($failCount -eq 0)
$outJsonFull = Join-Path $ProjectRoot $OutJson
$outMdFull = Join-Path $ProjectRoot $OutMd
Ensure-Dir -Path $outJsonFull
Ensure-Dir -Path $outMdFull

$payload = [ordered]@{
  schema = "p27_workflow_lint_report_v1"
  generated_at = (Get-Date).ToString("o")
  workflow_dir = $workflowRoot
  required_workflows = @($requiredFiles)
  missing_required = @($missingRequired)
  workflow_count = $workflowFiles.Count
  issue_count = $issueCount
  fail_count = $failCount
  pass = $overallPass
  reports = @($reports)
}
($payload | ConvertTo-Json -Depth 32) | Out-File -LiteralPath $outJsonFull -Encoding utf8

$mdLines = @(
  "# Workflow Lint Report (P27)",
  "",
  "- generated_at: " + $payload.generated_at,
  "- workflow_count: " + $payload.workflow_count,
  "- issue_count: " + $payload.issue_count,
  "- fail_count: " + $payload.fail_count,
  "- pass: " + $payload.pass,
  ""
)
if ($missingRequired.Count -gt 0) {
  $mdLines += "## Missing Required Workflows"
  foreach ($m in $missingRequired) { $mdLines += ("- " + $m) }
  $mdLines += ""
}
$mdLines += "## Per-Workflow Checks"
foreach ($r in $reports) {
  $mdLines += ("- " + $r.file + " | pass=" + $r.pass + " | issues=" + $r.issue_count)
  foreach ($i in $r.issues) {
    $mdLines += ("  - " + $i)
  }
}
$mdLines -join "`n" | Out-File -LiteralPath $outMdFull -Encoding utf8

($payload | ConvertTo-Json -Depth 32)
if (-not $overallPass) {
  exit 1
}
