@echo off
echo Astept inchiderea aplicatiei...
timeout /t 3 /nobreak >nul
"C:\Users\Gaby\Desktop\AI_Chess_Bot\.venv\Scripts\python.exe" -m pip uninstall -y torch
"C:\Users\Gaby\Desktop\AI_Chess_Bot\.venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
"C:\Users\Gaby\Desktop\AI_Chess_Bot\.venv\Scripts\python.exe" -c "import torch; print('CUDA disponibil:', torch.cuda.is_available())"
echo.
echo Daca mai sus scrie True: porneste din nou chess_bot.py. Daca a dat eroare sau scrie False,
echo incearca alta versiune CUDA din Setari, sau Python 3.12 (are cele mai multe pachete gata).
pause
