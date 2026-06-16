@echo off
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
set "CudaToolkitDir=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
set "PATH=%CUDA_PATH%\bin;%PATH%"
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
del /f /q "c:\Users\Siddh\OneDrive\Desktop\Projects\AES_GCM\build\CMakeCache.txt" 2>nul
cmake -S "c:\Users\Siddh\OneDrive\Desktop\Projects\AES_GCM" -B "c:\Users\Siddh\OneDrive\Desktop\Projects\AES_GCM\build" -G "Visual Studio 18 2026" -A x64 -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/nvcc.exe"
