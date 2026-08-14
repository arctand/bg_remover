# Background Remover

Локальное Windows-приложение для пакетного удаления фона с фотографий людей. Исходные
файлы обрабатываются локально и не отправляются во внешние API. Результат — RGBA PNG
исходного разрешения и с сохранением структуры подпапок.

## Требования

- Windows 11 x64 и Python 3.11/3.12 x64;
- NVIDIA GPU с CUDA (проверено на RTX 5070 12 GB);
- актуальный NVIDIA Driver;
- Git и интернет при первой установке пакетов и весов.

## Установка и запуск

1. Запустите `setup.bat`. Скрипт создаст `.venv`, установит PyTorch/CUDA и зависимости,
   скачает закреплённые веса production-моделей и выполнит smoke test.
2. Запустите `Background Remover.bat`.
3. Выберите папку с фотографиями и нажмите «Начать обработку».
4. При проблеме запустите `diagnose.bat` и сохраните вывод.

Приложение сохраняет результат рядом с исходной папкой:

```text
<source>_result/
  ready/          # автоматическая проверка не нашла сильных проблем
  review/         # PNG создан, но нужна ручная проверка
  failed/         # техническая ошибка обработки
  report.csv
  summary.json
```

Stop завершает текущий файл и останавливает очередь. Повторный запуск продолжает работу
по `report.csv`; исходники не перезаписываются. Resume пропускает только строки с тем же
`pipeline_fingerprint` и существующим output, поэтому legacy-отчёт или изменение модели/QC
автоматически вызывает повторную проверку файла. GUI намеренно показывает только выбор
папки, прогресс, Stop и историю, без внутренних настроек моделей/QC.

## Production pipeline

```text
SSDLite320 MobileNet V3 person detection (advisory, без crop)
→ full-frame ZhengPeng7/BiRefNet-portrait @ ecdeb6240ef23557dbd48ff27c59c1a88cbcb755
  (1024×1024, FP16 CUDA; alpha возвращается в исходное разрешение)
→ PyMatting 1.1.15 estimate_foreground_ml (только foreground RGB; alpha не меняется)
→ быстрый численный QC
→ facebook/sam2.1-hiera-small @ ee5bba1d82bb8749febdf90f45e84b687142ba03
  только для подозрительных случаев с уверенным person-box
→ READY / REVIEW / FAILED
```

SSDLite не обрезает кадр и не удаляет ничего за пределами bbox. Ноль детекций записывается
как `person_detector_zero=true`, но full-frame BiRefNet всё равно выполняется и результат
не становится FAILED/REVIEW только из-за детектора. `cropped_source_signal` также является
телеметрией: контакт объекта с рамкой сам по себе не переводит обычный портрет в REVIEW.

SAM 2.1 загружается лениво только после сильного сигнала: аномальной маски, слишком малого
покрытия person-box или сомнительного покрытия нескольких людей. Его prompted mask сравнивается
с alpha внутри соответствующего bbox с допуском на границы/волосы. SAM используется только как
независимый verifier и никогда не становится production alpha.

`ZhengPeng7/BiRefNet_HR-matting@5d6b6f8adcb5b417c871b1d84ceaae9871355b7f`
поддерживается классом backend как ручной fallback/диагностический вариант. Он не загружается
вместе с portrait. `BiRefNet_dynamic-matting` в production pipeline не используется.

## Диагностический benchmark

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe smoke_test.py
.venv\Scripts\python.exe benchmark.py "D:\Фото" "D:\Benchmark" --count 20
```

Последняя команда запускает production pipeline в test mode и создаёт previews на Original,
White, Gray, Black и Contrast фонах, contact sheet, `report.csv` и `summary.json` под
`debug_output/`. Фотографии и результаты исключены из git.

## Лицензии и commercial-use gate

**COMMERCIAL_LICENSE_REVIEW_REQUIRED:** на 14 августа 2026 года официальный репозиторий
BiRefNet содержит MIT License, но Hugging Face model card/repository
`ZhengPeng7/BiRefNet-portrait` не содержит отдельного явного license tag для checkpoint.
Нельзя считать коммерческую лицензию именно весов подтверждённой только по лицензии исходного
кода. Перед коммерческим выпуском необходимо получить/зафиксировать подтверждение владельца
весов. Это не блокирует локальный прототип.

- BiRefNet code: MIT — <https://github.com/ZhengPeng7/BiRefNet/blob/main/LICENSE>;
- portrait checkpoint/source: <https://huggingface.co/ZhengPeng7/BiRefNet-portrait>;
- PyMatting: MIT — <https://github.com/pymatting/pymatting>;
- SAM 2 code and checkpoints: Apache-2.0 — <https://github.com/facebookresearch/sam2>;
- TorchVision: BSD-style; Transformers/SAM dependency chain contains additional licenses;
- PySide6: LGPLv3/GPLv3/commercial. Проверяйте полный dependency/license inventory перед
  распространением сборки.

## Ограничения

- QC и SAM — эвристическая страховка, а не ground truth; REVIEW требует человека.
- Без уверенного SSD-Lite bbox SAM не запускается, поэтому архитектура/маленькие люди остаются
  зоной full-frame matting и численного QC.
- Приложение не выполняет generative fill, дорисовку тела, cloud inference или hard person crop.
