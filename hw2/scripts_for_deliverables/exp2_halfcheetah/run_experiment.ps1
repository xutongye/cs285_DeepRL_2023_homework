# CS285 HW2 Experiment 2 (HalfCheetah) - PG Algorithm Runner
# Run multiple PG algorithm experiments on HalfCheetah-v4 environment

$runHw2 = Join-Path $PSScriptRoot "..\..\cs285\scripts\run_hw2.py"
$runHw2 = (Resolve-Path $runHw2).Path

Write-Host "Starting CS285 HW2 Experiment 2 (HalfCheetah)..." -ForegroundColor Green

# Experiment 1: cheetah
Write-Host "1 Running experiment: halfcheetah" -ForegroundColor Yellow
python $runHw2 `
    --env_name HalfCheetah-v4 `
    -n 100 `
    -b 5000 `
    -eb 5000 `
    -rtg `
    --discount 0.95 `
    -lr 0.01 `
    --quiet `
    --exp_name halfcheetah

# Experiment 2: cheetah_baseline
Write-Host "2 Running experiment: halfcheetah_baseline" -ForegroundColor Yellow
python $runHw2 `
    --env_name HalfCheetah-v4 `
    -n 100 `
    -b 5000 `
    -eb 5000 `
    -rtg `
    --discount 0.95 `
    -lr 0.01 `
    --use_baseline `
    --quiet `
    --exp_name halfcheetah_baseline

# Experiment 3: cheetah_baseline_bgs2
Write-Host "3 Running experiment: halfcheetah_baseline_bgs2" -ForegroundColor Yellow
python $runHw2 `
    --env_name HalfCheetah-v4 `
    -n 100 `
    -b 5000 `
    -eb 5000 `
    --baseline_gradient_steps 2 `
    -rtg `
    --discount 0.95 `
    -lr 0.01 `
    --use_baseline `
    --quiet `
    --exp_name halfcheetah_baseline_bgs2

# Experiment 4: cheetah_baseline_nb
Write-Host "4 Running experiment: halfcheetah_baseline_nb" -ForegroundColor Yellow
python $runHw2 `
    --env_name HalfCheetah-v4 `
    -n 100 `
    -b 5000 `
    -eb 5000 `
    --baseline_gradient_steps 2 `
    -rtg `
    --discount 0.95 `
    -lr 0.01 `
    --use_baseline `
    --normalize_advantages `
    --quiet `
    --exp_name halfcheetah_baseline_nb

Write-Host "All experiments completed!" -ForegroundColor Green