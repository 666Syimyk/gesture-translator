$ErrorActionPreference = "Stop"

function Resolve-FullPath([string] $path) {
  return (Resolve-Path -LiteralPath $path).Path
}

$repoRoot = Resolve-FullPath (Join-Path $PSScriptRoot "..")
Set-Location -LiteralPath $repoRoot

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
  throw "git not found in PATH. Install Git and restart the terminal."
}

Write-Host "Building frontend..."
if (-not (Test-Path -LiteralPath "frontend/node_modules")) {
  npm --prefix frontend ci
}
npm --prefix frontend run build

$status = git status --porcelain
if ($status) {
  throw "Working directory is not clean. Commit or stash changes before deploy."
}

$distPath = Resolve-FullPath "frontend/dist"
$worktreeRel = ".gh-pages-worktree"
$worktreePath = Join-Path $repoRoot $worktreeRel

git worktree prune 2>$null | Out-Null
if (Test-Path -LiteralPath $worktreePath) {
  git worktree remove --force $worktreeRel 2>$null | Out-Null
  Remove-Item -LiteralPath $worktreePath -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "Preparing worktree for gh-pages..."
git fetch origin | Out-Null

$worktreeAdded = $false
try {
  $hasRemoteGhPages = (git ls-remote --exit-code --heads origin gh-pages 2>$null) -ne $null
  if ($hasRemoteGhPages) {
    git worktree add $worktreeRel gh-pages | Out-Null
  } else {
    git worktree add -b gh-pages $worktreeRel | Out-Null
  }
  $worktreeAdded = $true

  $worktreeFull = Resolve-FullPath $worktreeRel
  if (-not $worktreeFull.StartsWith($repoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Unsafe worktree path: $worktreeFull"
  }

  Write-Host "Updating gh-pages contents..."
  Get-ChildItem -LiteralPath $worktreeFull -Force |
    Where-Object { $_.Name -ne ".git" } |
    Remove-Item -Recurse -Force

  Copy-Item -Path (Join-Path $distPath "*") -Destination $worktreeFull -Recurse -Force
  Set-Content -LiteralPath (Join-Path $worktreeFull ".nojekyll") -Value "" -Encoding ASCII

  git -C $worktreeFull add -A
  $commitMsg = "Deploy to GitHub Pages ($(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))"
  git -C $worktreeFull commit -m $commitMsg | Out-Null
  git push origin gh-pages | Out-Null

  Write-Host "Done: gh-pages branch updated."
  Write-Host "In GitHub Pages settings select: Deploy from a branch -> gh-pages / (root)."
} finally {
  Set-Location -LiteralPath $repoRoot
  if ($worktreeAdded) {
    git worktree remove --force $worktreeRel 2>$null | Out-Null
  }
  Remove-Item -LiteralPath $worktreePath -Recurse -Force -ErrorAction SilentlyContinue
}

