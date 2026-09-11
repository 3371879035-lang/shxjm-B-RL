$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
python "$here\run_official.py" --mode 3 --strategy hybrid --model "$here\..\results\ppo_mode3_safe.pt" --log "$here\..\results\official_q3.jsonl"