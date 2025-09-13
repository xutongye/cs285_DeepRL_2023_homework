# CS285 HW2 Experiment 1 (CartPole) - PG Algorithm Runner
# Run multiple PG algorithm experiments on CartPole-v1 environment

Write-Host "Starting CS285 HW2 Experiment 1 (CartPole)..." -ForegroundColor Green

# Basic Experiment 1: cartpole
Write-Host "1 Running experiment: cartpole (baseline)" -ForegroundColor Yellow
python ../cs285/scripts/run_hw2.py --env_name CartPole-v1 -n 100 -b 1000 --exp_name cartpole --quiet

# Basic Experiment 2: cartpole_rtg (reward-to-go)
Write-Host "2 Running experiment: cartpole_rtg (reward-to-go)" -ForegroundColor Yellow
python ../cs285/scripts/run_hw2.py --env_name CartPole-v1 -n 100 -b 1000 -rtg --exp_name cartpole_rtg --quiet

# Basic Experiment 3: cartpole_na (normalized advantages)
Write-Host "3 Running experiment: cartpole_na (normalized advantages)" -ForegroundColor Yellow
python ../cs285/scripts/run_hw2.py --env_name CartPole-v1 -n 100 -b 1000 -na --exp_name cartpole_na --quiet

# Combined Experiment 4: cartpole_rtg_na (reward-to-go + normalized advantages)
Write-Host "4 Running experiment: cartpole_rtg_na (reward-to-go + normalized advantages)" -ForegroundColor Yellow
python ../cs285/scripts/run_hw2.py --env_name CartPole-v1 -n 100 -b 1000 -rtg -na --exp_name cartpole_rtg_na --quiet

# Large Batch Experiment 5: cartpole_lb (large batch)
Write-Host "5 Running experiment: cartpole_lb (large batch)" -ForegroundColor Yellow
python ../cs285/scripts/run_hw2.py --env_name CartPole-v1 -n 100 -b 4000 --exp_name cartpole_lb --quiet

# Large Batch Experiment 6: cartpole_lb_rtg (large batch + reward-to-go)
Write-Host "6 Running experiment: cartpole_lb_rtg (large batch + reward-to-go)" -ForegroundColor Yellow
python ../cs285/scripts/run_hw2.py --env_name CartPole-v1 -n 100 -b 4000 -rtg --exp_name cartpole_lb_rtg --quiet

# Large Batch Experiment 7: cartpole_lb_na (large batch + normalized advantages)
Write-Host "7 Running experiment: cartpole_lb_na (large batch + normalized advantages)" -ForegroundColor Yellow
python ../cs285/scripts/run_hw2.py --env_name CartPole-v1 -n 100 -b 4000 -na --exp_name cartpole_lb_na --quiet

# Complete Combined Experiment 8: cartpole_lb_rtg_na (large batch + reward-to-go + normalized advantages)
Write-Host "8 Running experiment: cartpole_lb_rtg_na (large batch + reward-to-go + normalized advantages)" -ForegroundColor Yellow
python ../cs285/scripts/run_hw2.py --env_name CartPole-v1 -n 100 -b 4000 -rtg -na --exp_name cartpole_lb_rtg_na --quiet

Write-Host "All experiments completed!" -ForegroundColor Green