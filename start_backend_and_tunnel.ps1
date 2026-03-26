<#
.SYNOPSIS
Starts the FastAPI Backend Server and Ngrok simultaneously in different windows.

.DESCRIPTION
This script will launch the WaferMap AI Backend on port 8000 and expose it via Ngrok to a secure HTTPs URL. 
Please ensure you have uvicorn and ngrok installed in your environment.
#>

Write-Host "============================" -ForegroundColor Cyan
Write-Host " WaferMap AI Tunnel Starter " -ForegroundColor Cyan
Write-Host "============================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting FastAPI server on Port 8000..."

# Start FastAPI process in a new window
# Note: assuming uvicorn is available via the pip requirements
Start-Process powershell -ArgumentList "-NoExit -Command `"uvicorn app:app --port 8000 --reload`""

Write-Host "Wait 3 seconds for server to spin up..."
Start-Sleep -Seconds 3

Write-Host "Starting Ngrok tunnel..."
# Start ngrok process in a new window
Start-Process powershell -ArgumentList "-NoExit -Command `"ngrok http 8000`""

Write-Host ""
Write-Host "✅ Backend & Tunnel started!" -ForegroundColor Green
Write-Host "Look at the cmd window that just popped up running 'ngrok' to find your Forwarding URL (e.g. https://xxxxxx.ngrok-free.app)."
Write-Host "Copy that URL and paste it into the frontend Configuration menu!"
Write-Host "Press any key to close this wrapper."
$Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
