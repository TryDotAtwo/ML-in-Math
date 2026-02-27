@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Target: notebook 89980 (v4 only), 91584 (beam only), baseline ~158680.
echo.
echo === 1) Evaluate baseline (reference) ===
python main.py evaluate --test baseline/sample_submission.csv --submission baseline/submission.csv
echo.
echo === 2) Solve: mode notebook (v4, goal 89980). Use --mode beam for 91584. ===
python run_best_score.py --mode notebook --out submission.csv
echo.
echo === 3) Evaluate new submission ===
python main.py evaluate --test baseline/sample_submission.csv --submission submission.csv
echo.
echo Done. submission.csv ready for Kaggle.
pause
