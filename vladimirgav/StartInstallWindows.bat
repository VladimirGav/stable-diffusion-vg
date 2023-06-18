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
set stable_diffusio_root=%cd%\..
set vladimirgav=%stable_diffusio_root%\vladimirgav
set sdvenv_path=%vladimirgav%\programs\sdvenv

set programsFolder=%vladimirgav%\programs
set pythonFolder=%vladimirgav%\programs\python
set pythonV=%vladimirgav%\programs\python\python
set pipPy=%vladimirgav%\programs\python\get-pip.py

REM set python_zip_url=https://www.python.org/ftp/python/3.10.0/python-3.10.0-embed-amd64.zip
set python_zip_url=https://vladimirgav.github.io/files/python/python-3.10.0-embed-amd64.zip
set python_archive_name=%vladimirgav%\programs\python-3.10.0-embed-amd64.zip

REM set pip_url=https://bootstrap.pypa.io/get-pip.py
set pip_url=https://vladimirgav.github.io/files/pip/get-pip.py

REM Create folder if it doesn't exist
if not exist %programsFolder% (
    MD %programsFolder%
	echo programsFolder created
)

REM python zip python and unzip in ./python
curl -o %python_archive_name% %python_zip_url%
powershell -Command "Expand-Archive -Path %python_archive_name% -DestinationPath %pythonFolder%"
del %python_archive_name%

REM copy files
copy %vladimirgav%\programs\python\python310._pth %vladimirgav%\programs\python\python310.pth
del %vladimirgav%\programs\python\python310._pth

REM add python in PATH
set PATH=%PATH%;%vladimirgav%\programs\python\
python -V

REM install pip
curl -o %pipPy% %pip_url%
python %pipPy% --force-reinstall
python -m pip list

REM install venv
python -m pip install virtualenv

REM create sdvenv
python -m virtualenv %sdvenv_path%

REM activate sdvenv
call %sdvenv_path%/Scripts/activate.bat

python -m pip list

REM pip install requirements
pip install -r %vladimirgav%\pipinstall\requirements.txt

REM Install Win Cuda 11.7.0 local https://developer.nvidia.com/cuda-11-7-0-download-archive
nvcc --version

REM pip instal
pip uninstall -y torch
pip install torch==1.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
ip uninstall -y torchvision
pip install torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip uninstall -y xformers
pip install xformers==0.0.16
pip uninstall -y invisible-watermark
pip install invisible-watermark

REM Install https://github.com/Stability-AI/stablediffusion or https://github.com/CompVis/stable-diffusion
REM python %stable_diffusio_root%\setup.py develop
REM cd %stable_diffusio_root%\
REM pip install -e .

pip list

REM Test
python %vladimirgav%\scripts\test_torch.py

REM Home models cache
set XDG_CACHE_HOME=%sdvenv_path%\cache

REM Create test image
REM python %vladimirgav%\scripts\vgTxt2img.py

REM pip list

pause