$ErrorActionPreference = "Continue"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $here
Write-Output "[driver] start mode3 100"
python "$here\auto_official_g25o_batch.py" --mode 3 --runs 100 --out-dir "$root\results\auto_g25o_p3_100_final"
Write-Output "[driver] start mode4 100"
python "$here\auto_official_g25o_batch.py" --mode 4 --runs 100 --out-dir "$root\results\auto_g25o_p4_100_final"
Write-Output "[driver] all done"