$ErrorActionPreference = "Stop"

function Resolve-FullPath([string] $path) {
  return (Resolve-Path -LiteralPath $path).Path
}

$repoRoot = Resolve-FullPath (Join-Path $PSScriptRoot "..")
Set-Location -LiteralPath $repoRoot

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
  throw "git не найден в PATH. Установите Git и перезапустите терминал."
}

if (-not (Test-Path -LiteralPath "frontend/dist")) {
  Write-Host "Собираю frontend..."
  npm --prefix frontend ci
  npm --prefix frontend run build
}

$status = git status --porcelain
if ($status) {
  throw "Рабочая директория не чистая. Закоммитьте или уберите изменения перед деплоем."
}

$distPath = Resolve-FullPath "frontend/dist"
$worktreeRel = ".gh-pages-worktree"
$worktreePath = Join-Path $repoRoot $worktreeRel

if (Test-Path -LiteralPath $worktreePath) {
  git worktree remove --force $worktreeRel | Out-Null
  Remove-Item -LiteralPath $worktreePath -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "Готовлю worktree для ветки gh-pages..."
git fetch origin | Out-Null

$hasRemoteGhPages = (git ls-remote --exit-code --heads origin gh-pages 2>$null) -ne $null
if ($hasRemoteGhPages) {
  git worktree add $worktreeRel gh-pages | Out-Null
} else {
  git worktree add -b gh-pages $worktreeRel | Out-Null
}

$worktreeFull = Resolve-FullPath $worktreeRel
if (-not $worktreeFull.StartsWith($repoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
  throw "Небезопасный путь worktree: $worktreeFull"
}

Write-Host "Очищаю содержимое ветки gh-pages и копирую сборку..."
Get-ChildItem -LiteralPath $worktreeFull -Force |
  Where-Object { $_.Name -ne ".git" } |
  Remove-Item -Recurse -Force

Copy-Item -Path (Join-Path $distPath "*") -Destination $worktreeFull -Recurse -Force

Set-Content -LiteralPath (Join-Path $worktreeFull ".nojekyll") -Value "" -Encoding ASCII

git -C $worktreeFull add -A
$commitMsg = "Deploy to GitHub Pages ($(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))"
git -C $worktreeFull commit -m $commitMsg | Out-Null
git push origin gh-pages | Out-Null

Write-Host "Готово: ветка gh-pages обновлена."
Write-Host "Теперь в настройках GitHub Pages выберите: Deploy from a branch -> gh-pages / (root)."

Set-Location -LiteralPath $repoRoot
git worktree remove --force $worktreeRel | Out-Null
Remove-Item -LiteralPath $worktreePath -Recurse -Force -ErrorAction SilentlyContinue

