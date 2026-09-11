@echo off
setlocal
cd /d C:\I2I
if not defined PUBLIC_BENCHMARK_ROOT set PUBLIC_BENCHMARK_ROOT=C:\I2I\public_benchmark_data
python seizure_detection\restore_public_event_annotations.py --dest "%PUBLIC_BENCHMARK_ROOT%"
if errorlevel 1 exit /b 1
python seizure_detection\research_campaign.py %*
exit /b %ERRORLEVEL%
