@echo off
echo Setting up LLVM environment for the compiler project...

REM Check if LLVM is already in PATH
where clang >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo LLVM is already in PATH.
) else (
    echo LLVM not found in PATH. Checking common installation locations...
    
    REM Check common installation locations
    set "FOUND=0"
    
    if exist "C:\Program Files\LLVM\bin\clang.exe" (
        set "LLVM_PATH=C:\Program Files\LLVM\bin"
        set "FOUND=1"
    ) else if exist "C:\Program Files (x86)\LLVM\bin\clang.exe" (
        set "LLVM_PATH=C:\Program Files (x86)\LLVM\bin"
        set "FOUND=1"
    ) else if exist "%USERPROFILE%\LLVM\bin\clang.exe" (
        set "LLVM_PATH=%USERPROFILE%\LLVM\bin"
        set "FOUND=1"
    )
    
    if "%FOUND%" == "1" (
        echo Found LLVM at: %LLVM_PATH%
        echo Adding LLVM to PATH...
        set "PATH=%LLVM_PATH%;%PATH%"
    ) else (
        echo LLVM not found. Running setup script to install LLVM...
        python setup_llvm.py
        
        REM Check if setup script created a batch file
        if exist "set_llvm_path.bat" (
            call set_llvm_path.bat
        )
    )
)

REM Run the specified Python script with arguments
if "%~1"=="" (
    echo Usage: run_with_llvm.bat [python_script.py] [arguments]
    echo Example: run_with_llvm.bat train_rl_agent.py --input-ir=test4.ll --episodes=1000
) else (
    echo Running: python %*
    python %*
)
