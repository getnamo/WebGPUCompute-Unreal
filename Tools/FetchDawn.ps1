# Downloads the pinned Dawn (WebGPU) prebuilt into ThirdParty/Dawn.
# Usage: powershell -ExecutionPolicy Bypass -File Tools/FetchDawn.ps1 [-Force]
#
# Dawn publishes nightly native builds on github.com/google/dawn/releases (best-effort, unsigned).
# We pin one release by tag + commit + SHA256 so builds are reproducible.

param(
	[switch]$Force
)

$ErrorActionPreference = 'Stop'

$DawnTag    = 'v20261009.190412'
$DawnCommit = 'c943ea32b614ff53e4fe410fc8f4968b6b8fe46d'
$Win64Asset = "Dawn-$DawnCommit-windows-latest-Release.tar.gz"
$Win64Sha256 = 'BA97FA93B5E1C265B8A4411698169B40E7696E7E8D574482C94EFCE6E3FF4228'

$PluginDir   = Split-Path -Parent $PSScriptRoot
$ThirdParty  = Join-Path $PluginDir 'ThirdParty'
$Downloads   = Join-Path $ThirdParty '.downloads'
$DawnDir     = Join-Path $ThirdParty 'Dawn'
$Marker      = Join-Path $DawnDir 'VERSION.txt'

if ((Test-Path $Marker) -and -not $Force)
{
	$Current = (Get-Content $Marker -Raw).Trim()
	if ($Current -eq "$DawnTag $DawnCommit")
	{
		Write-Host "Dawn $DawnTag already present in $DawnDir (use -Force to refetch)."
		exit 0
	}
}

New-Item -ItemType Directory -Force $Downloads | Out-Null

$Archive = Join-Path $Downloads $Win64Asset
if (-not (Test-Path $Archive))
{
	$Url = "https://github.com/google/dawn/releases/download/$DawnTag/$Win64Asset"
	Write-Host "Downloading $Url"
	Invoke-WebRequest -Uri $Url -OutFile $Archive -UseBasicParsing
}

# .NET hashing instead of Get-FileHash: Windows PowerShell launched from pwsh 7 can fail to load that cmdlet.
$Stream = [System.IO.File]::OpenRead($Archive)
try { $Hash = [System.BitConverter]::ToString([System.Security.Cryptography.SHA256]::Create().ComputeHash($Stream)).Replace('-', '') }
finally { $Stream.Dispose() }
if ($Hash -ne $Win64Sha256)
{
	Remove-Item $Archive -Force
	throw "SHA256 mismatch for $Win64Asset (got $Hash, expected $Win64Sha256)"
}

# Extract into a scratch folder, then copy only what the plugin uses.
$Extract = Join-Path $Downloads 'extract'
if (Test-Path $Extract) { Remove-Item $Extract -Recurse -Force }
New-Item -ItemType Directory -Force $Extract | Out-Null
# System32 tar: Git's GNU tar misreads drive letters.
& "$env:SystemRoot\System32\tar.exe" -xzf $Archive -C $Extract
if ($LASTEXITCODE -ne 0) { throw "tar failed with $LASTEXITCODE" }

$Root = Join-Path $Extract "Dawn-$DawnCommit-windows-latest-Release"

if (Test-Path $DawnDir) { Remove-Item $DawnDir -Recurse -Force }
New-Item -ItemType Directory -Force (Join-Path $DawnDir 'include\dawn\native'), (Join-Path $DawnDir 'include\webgpu'), (Join-Path $DawnDir 'lib\Win64') | Out-Null

Copy-Item (Join-Path $Root 'include\webgpu\*') (Join-Path $DawnDir 'include\webgpu')
Get-ChildItem (Join-Path $Root 'include\dawn') -File | Copy-Item -Destination (Join-Path $DawnDir 'include\dawn')
Copy-Item (Join-Path $Root 'include\dawn\native\*') (Join-Path $DawnDir 'include\dawn\native')
Copy-Item (Join-Path $Root 'lib\webgpu_dawn.lib') (Join-Path $DawnDir 'lib\Win64')

Invoke-WebRequest -Uri "https://raw.githubusercontent.com/google/dawn/$DawnCommit/LICENSE" -OutFile (Join-Path $DawnDir 'LICENSE') -UseBasicParsing

Set-Content -Path $Marker -Value "$DawnTag $DawnCommit"
Remove-Item $Extract -Recurse -Force

Write-Host "Dawn $DawnTag ($DawnCommit) installed to $DawnDir"
