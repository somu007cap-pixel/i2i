@echo off
setlocal

cd /d "%~dp0"

if not exist "showcase_outputs\logs" mkdir "showcase_outputs\logs"
if not exist "seizure_detection\outputs\marathon" mkdir "seizure_detection\outputs\marathon"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set RUN_TS=%%i
set LOG_FILE=%CD%\showcase_outputs\logs\marathon_%RUN_TS%.log
set STATUS_FILE=%CD%\seizure_detection\outputs\marathon\marathon_%RUN_TS%_status.txt

set PYTHONUTF8=1
set TF_CPP_MIN_LOG_LEVEL=3
set TF_ENABLE_ONEDNN_OPTS=0

if not defined MARATHON_MAX_SESSIONS set MARATHON_MAX_SESSIONS=146
if not defined MARATHON_TSMIXER_EPOCHS set MARATHON_TSMIXER_EPOCHS=24
if not defined MARATHON_TSMIXER_BATCH_SIZE set MARATHON_TSMIXER_BATCH_SIZE=64
if not defined MARATHON_TSMIXER_MAX_TRAIN_WINDOWS set MARATHON_TSMIXER_MAX_TRAIN_WINDOWS=600000
if not defined MARATHON_TSMIXER_PATIENCE set MARATHON_TSMIXER_PATIENCE=7
if not defined MARATHON_BASELINE_EPOCHS set MARATHON_BASELINE_EPOCHS=12
if not defined MARATHON_BASELINE_BATCH_SIZE set MARATHON_BASELINE_BATCH_SIZE=128
if not defined MARATHON_BASELINE_MAX_TRAIN_WINDOWS set MARATHON_BASELINE_MAX_TRAIN_WINDOWS=250000
if not defined MARATHON_BASELINE_AE_MAX_NORMAL_WINDOWS set MARATHON_BASELINE_AE_MAX_NORMAL_WINDOWS=250000
if not defined MARATHON_EDGE_BENCHMARK set MARATHON_EDGE_BENCHMARK=1
if not defined MARATHON_EDGE_BENCHMARK_RUNS set MARATHON_EDGE_BENCHMARK_RUNS=50
if not defined MARATHON_EDGE_BENCHMARK_WARMUP set MARATHON_EDGE_BENCHMARK_WARMUP=5
if not defined MARATHON_EDGE_FAIL_ON_BENCHMARK_FAILURE set MARATHON_EDGE_FAIL_ON_BENCHMARK_FAILURE=0
if not defined MARATHON_EDGE_REQUIRE_INT8 set MARATHON_EDGE_REQUIRE_INT8=0
if not defined MARATHON_FINAL_FOCUS set MARATHON_FINAL_FOCUS=1
if not defined MARATHON_ROBUSTNESS_STUDY set MARATHON_ROBUSTNESS_STUDY=0
if not defined MARATHON_CLEAN_OUTPUTS set MARATHON_CLEAN_OUTPUTS=1
if not defined MARATHON_RESUME set MARATHON_RESUME=0
if /I "%~1"=="--resume" set MARATHON_RESUME=1
if /I "%~2"=="--resume" set MARATHON_RESUME=1
set MARATHON_CHECK=0
if /I "%~1"=="--check" set MARATHON_CHECK=1
if /I "%~2"=="--check" set MARATHON_CHECK=1
set MARATHON_RESUME_FLAG=
if "%MARATHON_RESUME%"=="1" (
  set MARATHON_CLEAN_OUTPUTS=0
  set MARATHON_RESUME_FLAG=--resume
)

if "%MARATHON_CHECK%"=="1" (
  echo Check mode only. No training will be started.
  echo Python: %CD%\.venv\Scripts\python.exe
  echo Log file would be: %LOG_FILE%
  echo Status file would be: %STATUS_FILE%
  echo MARATHON_MAX_SESSIONS=%MARATHON_MAX_SESSIONS%
  echo MARATHON_TSMIXER_EPOCHS=%MARATHON_TSMIXER_EPOCHS%
  echo MARATHON_TSMIXER_BATCH_SIZE=%MARATHON_TSMIXER_BATCH_SIZE%
  echo MARATHON_TSMIXER_MAX_TRAIN_WINDOWS=%MARATHON_TSMIXER_MAX_TRAIN_WINDOWS%
  echo MARATHON_BASELINE_EPOCHS=%MARATHON_BASELINE_EPOCHS%
  echo MARATHON_BASELINE_BATCH_SIZE=%MARATHON_BASELINE_BATCH_SIZE%
  echo MARATHON_EDGE_BENCHMARK=%MARATHON_EDGE_BENCHMARK%
  echo MARATHON_EDGE_BENCHMARK_RUNS=%MARATHON_EDGE_BENCHMARK_RUNS%
  echo MARATHON_EDGE_BENCHMARK_WARMUP=%MARATHON_EDGE_BENCHMARK_WARMUP%
  echo MARATHON_EDGE_FAIL_ON_BENCHMARK_FAILURE=%MARATHON_EDGE_FAIL_ON_BENCHMARK_FAILURE%
  echo MARATHON_EDGE_REQUIRE_INT8=%MARATHON_EDGE_REQUIRE_INT8%
  echo MARATHON_FINAL_FOCUS=%MARATHON_FINAL_FOCUS%
  echo MARATHON_ROBUSTNESS_STUDY=%MARATHON_ROBUSTNESS_STUDY%
  echo MARATHON_CLEAN_OUTPUTS=%MARATHON_CLEAN_OUTPUTS%
  echo MARATHON_RESUME=%MARATHON_RESUME%
  if not exist "%CD%\.venv\Scripts\python.exe" (
    echo ERROR: project venv python not found.
    exit /b 1
  )
  exit /b 0
)

if "%MARATHON_CLEAN_OUTPUTS%"=="1" (
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ErrorActionPreference='Stop'; $root=(Resolve-Path '%CD%').Path; $targets=@('%CD%\seizure_detection\outputs','%CD%\prediction_outputs_local','%CD%\showcase_outputs'); foreach($target in $targets){ if(Test-Path -LiteralPath $target){ $resolved=(Resolve-Path -LiteralPath $target).Path; if(-not $resolved.StartsWith($root,[System.StringComparison]::OrdinalIgnoreCase)){ throw ('Refusing to clean outside workspace: ' + $resolved) }; Get-ChildItem -LiteralPath $resolved -Force | Remove-Item -Recurse -Force } else { New-Item -ItemType Directory -Path $target | Out-Null } }; New-Item -ItemType Directory -Force -Path '%CD%\seizure_detection\outputs\marathon','%CD%\showcase_outputs\logs','%CD%\prediction_outputs_local' | Out-Null"
  if errorlevel 1 exit /b 1
)

echo ===============================================================
echo MARATHON experiments started at %DATE% %TIME%
echo Working directory: %CD%
echo Log file: %LOG_FILE%
echo Status file: %STATUS_FILE%
echo ===============================================================

(
  echo MARATHON START %DATE% %TIME%
  echo MAX_SESSIONS=%MARATHON_MAX_SESSIONS%
  echo TSMIXER_EPOCHS=%MARATHON_TSMIXER_EPOCHS%
  echo BASELINE_EPOCHS=%MARATHON_BASELINE_EPOCHS%
  echo EDGE_BENCHMARK=%MARATHON_EDGE_BENCHMARK%
  echo EDGE_BENCHMARK_RUNS=%MARATHON_EDGE_BENCHMARK_RUNS%
  echo EDGE_BENCHMARK_WARMUP=%MARATHON_EDGE_BENCHMARK_WARMUP%
  echo EDGE_FAIL_ON_BENCHMARK_FAILURE=%MARATHON_EDGE_FAIL_ON_BENCHMARK_FAILURE%
  echo EDGE_REQUIRE_INT8=%MARATHON_EDGE_REQUIRE_INT8%
  echo MARATHON_FINAL_FOCUS=%MARATHON_FINAL_FOCUS%
  echo MARATHON_ROBUSTNESS_STUDY=%MARATHON_ROBUSTNESS_STUDY%
  echo MARATHON_RESUME=%MARATHON_RESUME%
) > "%STATUS_FILE%"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "& { $ErrorActionPreference='Continue';" ^
  "$env:PYTHONUTF8='1'; $env:TF_CPP_MIN_LOG_LEVEL='3'; $env:TF_ENABLE_ONEDNN_OPTS='0'; $env:EDGE_BENCHMARK='%MARATHON_EDGE_BENCHMARK%'; $env:EDGE_BENCHMARK_RUNS='%MARATHON_EDGE_BENCHMARK_RUNS%'; $env:EDGE_BENCHMARK_WARMUP='%MARATHON_EDGE_BENCHMARK_WARMUP%'; $env:EDGE_FAIL_ON_BENCHMARK_FAILURE='%MARATHON_EDGE_FAIL_ON_BENCHMARK_FAILURE%'; $env:EDGE_REQUIRE_INT8='%MARATHON_EDGE_REQUIRE_INT8%'; $env:TSMIXER_STOP_ON_FAILURE='1';" ^
  "Write-Host '===============================================================';" ^
  "Write-Host '[0/9] Preflight checks';" ^
  "Write-Host '===============================================================';" ^
  "Add-Content -Path '%STATUS_FILE%' -Value ('START PREFLIGHT ' + (Get-Date -Format s));" ^
  "& '%CD%\.venv\Scripts\python.exe' '%CD%\seizure_detection\preflight_check.py';" ^
  "$preCode=$LASTEXITCODE; Add-Content -Path '%STATUS_FILE%' -Value ('END PREFLIGHT code=' + $preCode + ' ' + (Get-Date -Format s)); if ($preCode -ne 0) { exit $preCode };" ^
  "Write-Host '===============================================================';" ^
  "Write-Host '[1/9] Label metadata audit';" ^
  "Write-Host '===============================================================';" ^
  "Add-Content -Path '%STATUS_FILE%' -Value ('START LABEL AUDIT ' + (Get-Date -Format s));" ^
  "& '%CD%\.venv\Scripts\python.exe' '%CD%\seizure_detection\label_metadata_audit.py';" ^
  "$labelCode=$LASTEXITCODE; Add-Content -Path '%STATUS_FILE%' -Value ('END LABEL AUDIT code=' + $labelCode + ' ' + (Get-Date -Format s)); if ($labelCode -ne 0) { exit $labelCode };" ^
  "Write-Host '===============================================================';" ^
  "Write-Host '[2/9] Patched Dual-Stream TSMixer experiment sweep';" ^
  "Write-Host '===============================================================';" ^
  "Add-Content -Path '%STATUS_FILE%' -Value ('START TSMIXER ' + (Get-Date -Format s));" ^
  "if ('%MARATHON_ROBUSTNESS_STUDY%' -eq '1') { $tsmixerArgs = @('--robust-final') } elseif ('%MARATHON_FINAL_FOCUS%' -eq '1') { $tsmixerArgs = @('--focus-final') } else { $tsmixerArgs = @() };" ^
  "& '%CD%\.venv\Scripts\python.exe' '%CD%\seizure_detection\run_tsmixer_experiments.py' --max-sessions %MARATHON_MAX_SESSIONS% --epochs %MARATHON_TSMIXER_EPOCHS% --batch-size %MARATHON_TSMIXER_BATCH_SIZE% --max-train-windows %MARATHON_TSMIXER_MAX_TRAIN_WINDOWS% --patience %MARATHON_TSMIXER_PATIENCE% %MARATHON_RESUME_FLAG% $tsmixerArgs;" ^
  "$tsCode=$LASTEXITCODE; Add-Content -Path '%STATUS_FILE%' -Value ('END TSMIXER code=' + $tsCode + ' ' + (Get-Date -Format s)); if ($tsCode -ne 0) { exit $tsCode };" ^
  "Write-Host '===============================================================';" ^
  "Write-Host '[3/9] Promote best TSMixer experiment';" ^
  "Write-Host '===============================================================';" ^
  "Add-Content -Path '%STATUS_FILE%' -Value ('START PROMOTE ' + (Get-Date -Format s));" ^
  "& '%CD%\.venv\Scripts\python.exe' '%CD%\seizure_detection\promote_best_tsmixer_experiment.py';" ^
  "$promoteCode=$LASTEXITCODE; Add-Content -Path '%STATUS_FILE%' -Value ('END PROMOTE code=' + $promoteCode + ' ' + (Get-Date -Format s)); if ($promoteCode -ne 0) { exit $promoteCode };" ^
  "Write-Host '===============================================================';" ^
  "Write-Host '[4/9] Refresh TSMixer experiment summary';" ^
  "Write-Host '===============================================================';" ^
  "& '%CD%\.venv\Scripts\python.exe' '%CD%\seizure_detection\summarize_tsmixer_experiments.py';" ^
  "$sumCode=$LASTEXITCODE; Add-Content -Path '%STATUS_FILE%' -Value ('END TSMIXER SUMMARY code=' + $sumCode + ' ' + (Get-Date -Format s)); if ($sumCode -ne 0) { exit $sumCode };" ^
  "Write-Host '===============================================================';" ^
  "Write-Host '[5/9] Baseline comparison suite';" ^
  "Write-Host '===============================================================';" ^
  "Add-Content -Path '%STATUS_FILE%' -Value ('START BASELINES ' + (Get-Date -Format s));" ^
  "$env:DETECTION_MAX_SESSIONS='%MARATHON_MAX_SESSIONS%'; $env:BASELINE_EPOCHS='%MARATHON_BASELINE_EPOCHS%'; $env:BASELINE_BATCH_SIZE='%MARATHON_BASELINE_BATCH_SIZE%'; $env:BASELINE_MAX_TRAIN_WINDOWS='%MARATHON_BASELINE_MAX_TRAIN_WINDOWS%'; $env:BASELINE_AE_MAX_NORMAL_WINDOWS='%MARATHON_BASELINE_AE_MAX_NORMAL_WINDOWS%';" ^
  "& '%CD%\.venv\Scripts\python.exe' '%CD%\seizure_detection\detection_baselines.py';" ^
  "$baseCode=$LASTEXITCODE; Add-Content -Path '%STATUS_FILE%' -Value ('END BASELINES code=' + $baseCode + ' ' + (Get-Date -Format s)); if ($baseCode -ne 0) { exit $baseCode };" ^
  "Write-Host '===============================================================';" ^
  "Write-Host '[6/9] Modern method coverage report';" ^
  "Write-Host '===============================================================';" ^
  "Add-Content -Path '%STATUS_FILE%' -Value ('START METHOD COVERAGE ' + (Get-Date -Format s));" ^
  "& '%CD%\.venv\Scripts\python.exe' '%CD%\seizure_detection\method_coverage_report.py';" ^
  "$methodCode=$LASTEXITCODE; Add-Content -Path '%STATUS_FILE%' -Value ('END METHOD COVERAGE code=' + $methodCode + ' ' + (Get-Date -Format s)); if ($methodCode -ne 0) { exit $methodCode };" ^
  "Write-Host '===============================================================';" ^
  "Write-Host '[7/9] Final edge feasibility refresh';" ^
  "Write-Host '===============================================================';" ^
  "Add-Content -Path '%STATUS_FILE%' -Value ('START EDGE ' + (Get-Date -Format s));" ^
  "& '%CD%\.venv\Scripts\python.exe' '%CD%\seizure_detection\edge_feasibility.py';" ^
  "$edgeCode=$LASTEXITCODE; Add-Content -Path '%STATUS_FILE%' -Value ('END EDGE code=' + $edgeCode + ' ' + (Get-Date -Format s)); if ($edgeCode -ne 0) { exit $edgeCode };" ^
  "Write-Host '===============================================================';" ^
  "Write-Host '[8/9] Final study report';" ^
  "Write-Host '===============================================================';" ^
  "Add-Content -Path '%STATUS_FILE%' -Value ('START FINAL STUDY REPORT ' + (Get-Date -Format s));" ^
  "& '%CD%\.venv\Scripts\python.exe' '%CD%\seizure_detection\final_study_report.py';" ^
  "$reportCode=$LASTEXITCODE; Add-Content -Path '%STATUS_FILE%' -Value ('END FINAL STUDY REPORT code=' + $reportCode + ' ' + (Get-Date -Format s)); if ($reportCode -ne 0) { exit $reportCode };" ^
  "Write-Host '===============================================================';" ^
  "Write-Host '[9/9] Dissertation figures';" ^
  "Write-Host '===============================================================';" ^
  "Add-Content -Path '%STATUS_FILE%' -Value ('START DISSERTATION FIGURES ' + (Get-Date -Format s));" ^
  "& '%CD%\.venv\Scripts\python.exe' '%CD%\seizure_detection\dissertation_visuals.py';" ^
  "$figureCode=$LASTEXITCODE; Add-Content -Path '%STATUS_FILE%' -Value ('END DISSERTATION FIGURES code=' + $figureCode + ' ' + (Get-Date -Format s)); if ($figureCode -ne 0) { exit $figureCode };" ^
  "Add-Content -Path '%STATUS_FILE%' -Value ('MARATHON COMPLETE ' + (Get-Date -Format s));" ^
  "exit 0 }" ^
  2^>^&1 | powershell -NoProfile -Command "$input | Tee-Object -FilePath '%LOG_FILE%'"

set EXIT_CODE=%ERRORLEVEL%

echo ===============================================================
echo MARATHON experiments finished at %DATE% %TIME%
echo Exit code: %EXIT_CODE%
echo ===============================================================
echo Main outputs:
echo   showcase_outputs\logs\marathon_%RUN_TS%.log
echo   seizure_detection\outputs\marathon\marathon_%RUN_TS%_status.txt
echo   seizure_detection\outputs\tsmixer_experiments\TSMIXER_EXPERIMENT_SUMMARY.md
echo   seizure_detection\outputs\baselines\BASELINE_COMPARISON_SUMMARY.md
echo   seizure_detection\outputs\SOTA_METHOD_COVERAGE.md
echo   seizure_detection\outputs\edge_feasibility\EDGE_FEASIBILITY_SUMMARY.md
echo   seizure_detection\outputs\FINAL_STUDY_REPORT.md
echo   seizure_detection\outputs\dissertation_figures\README_FIGURES.md
echo ===============================================================

pause
exit /b %EXIT_CODE%
