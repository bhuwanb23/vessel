@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64 >nul 2>&1
cd /d D:\projects\software\local_llm
if exist build_plan\CMakeCache.txt del build_plan\CMakeCache.txt
if exist build_plan\CMakeFiles rmdir /s /q build_plan\CMakeFiles
"C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" -B build_plan -S . -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
if %ERRORLEVEL% neq 0 (
    echo CMake configure failed!
    exit /b 1
)
"C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build_plan --config Release
if %ERRORLEVEL% neq 0 (
    echo Build failed!
    exit /b 1
)
echo === Build successful! ===
build_plan\planner.exe
