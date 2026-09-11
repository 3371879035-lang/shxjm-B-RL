$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
python "$here\run_official.py" --mode 4 --strategy hybrid --model "$here\..\results\ppo_mode4_safe.pt" --log "$here\..\results\official_q4.jsonl"