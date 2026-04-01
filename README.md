# ClearML Learning

PyTorch Lightning + Hydra 기반 ML 학습 프로젝트.

## 프로젝트 구조

```
├── configs/                  # Hydra 설정 파일
│   ├── callbacks/            # 콜백 설정
│   ├── data/                 # 데이터 설정
│   ├── debug/                # 디버깅 설정
│   ├── experiment/           # 실험 설정
│   ├── extras/               # 부가 기능 설정
│   ├── hparams_search/       # 하이퍼파라미터 탐색 설정
│   ├── hydra/                # Hydra 런타임 설정
│   ├── logger/               # 로거 설정
│   ├── model/                # 모델 설정
│   ├── paths/                # 경로 설정
│   ├── trainer/              # Trainer 설정
│   ├── train.yaml            # 학습 기본 설정
│   └── eval.yaml             # 평가 기본 설정
├── src/
│   ├── data/                 # DataModule
│   ├── models/               # LightningModule + 모델 컴포넌트
│   ├── utils/                # 유틸리티
│   ├── train.py              # 학습 엔트리포인트
│   └── eval.py               # 평가 엔트리포인트
├── tests/                    # 테스트
├── scripts/
│   ├── clearml_run.py        # ClearML 원격 실행 스크립트
│   └── schedule.sh           # 배치 학습 스크립트
├── notebooks/                # 노트북
├── pyproject.toml            # 프로젝트 설정 및 의존성
└── .python-version           # Python 버전 (3.10)
```

## 환경 설정

[uv](https://docs.astral.sh/uv/)를 사용합니다.

```bash
# 의존성 설치
uv sync
```

## 학습

```bash
# 기본 설정으로 학습
uv run python src/train.py

# 설정 오버라이드
uv run python src/train.py trainer.max_epochs=20 data.batch_size=64

# GPU 사용
uv run python src/train.py trainer=gpu

# 실험 설정 사용
uv run python src/train.py experiment=example
```

## 평가

```bash
uv run python src/eval.py ckpt_path="logs/train/runs/.../checkpoints/epoch_xxx.ckpt"
```

## ClearML 원격 실행

`scripts/clearml_run.py`를 통해 기존 학습 코드를 수정하지 않고 ClearML Agent에 태스크를 전달할 수 있습니다.

### 사전 준비

```bash
# ClearML 인증 설정
uv run clearml-init
```

### 사용법

```bash
# 기본 실행 (src/train.py를 ClearML 태스크로 enqueue)
uv run python scripts/clearml_run.py

# Hydra 인자 전달 (기존 학습 명령과 동일한 형태)
uv run python scripts/clearml_run.py trainer.max_epochs=5 logger=csv

# Docker 이미지 지정
uv run python scripts/clearml_run.py --docker clearml-worker:latest trainer.max_epochs=10

# 큐 지정
uv run python scripts/clearml_run.py --queue gpu_queue trainer=gpu

# 기존 태스크를 복제하여 다른 파라미터로 재실행
uv run python scripts/clearml_run.py --clone <task_id> trainer.max_epochs=20
```

### 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--project` | `ClearML-Learning` | ClearML 프로젝트 이름 |
| `--name` | `train` | 태스크 이름 |
| `--script` | `src/train.py` | 실행할 스크립트 |
| `--queue` | `default` | ClearML 큐 이름 |
| `--clone` | - | 복제할 기존 태스크 ID |
| `--docker` | - | Agent가 사용할 Docker 이미지 |

Hydra 오버라이드는 옵션 뒤에 그대로 붙이면 됩니다. 전달된 인자는 `Task.set_parameters()`를 통해 `Args/` 네임스페이스 아래에 등록됩니다(예: `trainer.max_epochs=5` → `Args/trainer.max_epochs = 5`).

## 테스트

```bash
uv run pytest
```

## 기술 스택

- **Python** 3.10
- **PyTorch** 2.5.*
- **Lightning** 2.x
- **Hydra** 1.3.2
- **ClearML** 2.1.5+
- **TensorBoard** 2.20+
