# Git Sync And Safe Branch Cleanup

This document describes how to use `scripts/git_sync.ps1` for repeatable branch sync and safe local cleanup.

## Prerequisites

1. Repository has a configured remote (usually `origin`).
2. You are inside the repository root.
3. Run dry-run first.

If `origin` does not exist, add it first, for example:

```powershell
git remote add origin <repo_url>
```

## Script

`powershell -ExecutionPolicy Bypass -File scripts\git_sync.ps1`

## Common examples

### 1) Dry-run (default, recommended)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\git_sync.ps1 -DryRun:$true
```

### 2) Real execution (push + cleanup)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\git_sync.ps1 -DryRun:$false
```

### 3) Only clean gone branches (no merged cleanup)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\git_sync.ps1 -DryRun:$false -DeleteMerged:$false -DeleteGone:$true
```

### 4) Push only, no branch cleanup

```powershell
powershell -ExecutionPolicy Bypass -File scripts\git_sync.ps1 -DryRun:$false -DeleteMerged:$false -DeleteGone:$false
```

### 5) Force delete local branches (dangerous)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\git_sync.ps1 -DryRun:$false -Force:$true
```

## Safety behavior

1. Dry-run is enabled by default.
2. Protected branches are never deleted (`main`, `master`, `dev`, `develop`, `release/*`, `hotfix/*`, current branch).
3. Pull uses `--ff-only`.
4. If remote is missing, push/fetch are skipped and local cleanup is dry-run only unless `-ForceLocalCleanup:$true` is passed.

## Reports

Every run writes a machine-readable report:

`docs/artifacts/git_sync/git_sync_<timestamp>.json`

The report includes params, pre-state, planned actions, executed actions, and summary.
