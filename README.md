# Stable Diffusion для Windows от VladimirGav

## Официальные репозитории
- https://github.com/Stability-AI/stablediffusion
- https://github.com/CompVis/stable-diffusion

## Возможности сборки
- Автоматическая установка для Windows
- Генерация изображений по текстовому описанию
- Генерация изображений по изображению и текстовому описанию
- Подключение к телеграм боту https://github.com/VladimirGav/telegrambot
- Автоматическая загрузка и обновление моделей из https://huggingface.co/
- Работает со слабыми видеокартами, протестировал на NVIDIA GeForce RTX 3050 Laptop GPU

## Установка на Windows
Видео инструкция https://youtu.be/dUGForWid64
1. Установите Windows Cuda local версии 11.7.0 для своей видеокарты
- Для NVIDIA https://developer.nvidia.com/cuda-11-7-0-download-archive
2. Загружаем zip архив репозитория VladimirGav/stable-diffusion-vg на компьютер и распаковываем.
3. Запускаем автоматическую установку на Windows [vladimirgav/StartInstallWindows.bat](vladimirgav/StartInstallWindows.bat)

## Как генерировать изображения по текстовому описанию в Windows
Видео инструкция скоро
1. Вводим `модель, ширину, высоту, описание, отрицание и другое` в файл [vladimirgav/inputdata/txt2img.json](vladimirgav/inputdata/txt2img.json)
2. Запускаем файл [vladimirgav/vg_txt2img.bat](vladimirgav/vg_txt2img.bat)
3. Получаем готовое изображение, например в `vladimirgav/imgs`

## Как генерировать изображения по изображению и текстовому описанию в Windows
Видео инструкция скоро
1. Помещаем исходное изображение в [vladimirgav/inputdata/img2img.jpg](vladimirgav/inputdata/img2img.jpg) и вводим `модель, описание, отрицание и другое` в файл [vladimirgav/inputdata/img2img.json](vladimirgav/inputdata/img2img.json)
2. Запускаем файл [vladimirgav/vg_img2img.bat](vladimirgav/vg_img2img.bat)
3. Получаем готовое изображение, например в `vladimirgav/imgs`

Разработчик: VladimirGav