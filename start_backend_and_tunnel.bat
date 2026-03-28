@echo off
echo ============================
echo  WaferMap AI Tunnel Starter
echo ============================
echo.
echo Starting FastAPI server on Port 8000...
start "FastAPI Server" cmd /k "python -m uvicorn app:app --port 8000 --reload"

echo Wait 3 seconds for server to spin up...
timeout /t 3 /nobreak >nul

echo Starting Ngrok tunnel...
start "Ngrok Tunnel" cmd /k "ngrok http 8000"

echo.
echo ✅ Backend ^& Tunnel started!
echo Look at the cmd window that just popped up running 'ngrok' to find your Forwarding URL (e.g. https://xxxxxx.ngrok-free.app).
echo Copy that URL and paste it into the frontend Configuration menu!
echo Press any key to close this wrapper.
pause >nul
