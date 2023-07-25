@echo off

goto start
--------------------------------------
VladimirGav
GitHub Website: https://vladimirgav.github.io/
GitHub: https://github.com/VladimirGav
Source https://github.com/VladimirGav/stable-diffusion-vg
Copyright (c)
--------------------------------------
:start

REM path
set vladimirgav=%cd%
set sdvenv_path=%vladimirgav%\programs\sdvenv

set py_path=%vladimirgav%\programs\python\
set PATH=%py_path%;%PATH%

REM activate sdvenv
REM call %sdvenv_path%/Scripts/activate.bat

python --version

pause