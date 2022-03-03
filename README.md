# YOLOv5 Installable Package
Copy of [YOLOv5 repository](https://github.com/ultralytics/yolov5) wrapped as an installable pip package.

Inspired from https://github.com/fcakyon/yolov5-pip.


## Build Package
Build wheel file:
```bash
python setup.py bdist_wheel
```


## Issues to resolve when cloning `yolov5` repository
Integrating `yolov5` repository as an installable package brings import issues.
This section covers parts of code that were modified.

### General imports
Insert `yolov5.` prefix when importing modules `models` and `utils`.
For example: 
```python
from yolov5.models.common import DetectMultiBackend
```

### Wandb
Directory `yolov5.utils.loggers` contains custom module `wandb` which has the same name as an installable pip package `wandb`.
To prevent import errors the custom module `yolov5.utils.loggers.wandb` was renamed to `yolov5.utils.loggers.wandb_`.


## Authors
**Ray Chamidullin** - chamidullinr@gmail.com  - [Github account](https://github.com/chamidullinr)
