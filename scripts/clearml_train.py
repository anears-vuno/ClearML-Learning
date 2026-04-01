"""ClearML Agent용 학습 래퍼.

ClearML Agent가 이 스크립트를 entry point로 실행한다.
Task.init()으로 ClearML 추적을 활성화한 뒤, src/train.py를 __main__으로 실행하여
Hydra의 config_path가 파일 시스템 경로로 정상 해석되도록 한다.
"""

import runpy

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from clearml import Task

Task.init()

runpy.run_path("src/train.py", run_name="__main__")
