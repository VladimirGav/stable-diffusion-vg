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

set PATH=%PATH%;%vladimirgav%\programs\python\

REM activate sdvenv
call %sdvenv_path%/Scripts/activate.bat

REM Home models cache
set XDG_CACHE_HOME=%sdvenv_path%\cache

REM Create test image
python %vladimirgav%\scripts\vgImg2img.py

pause