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
├── scripts/                  # 실행 스크립트
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

## 테스트

```bash
uv run pytest
```

## 기술 스택

- **Python** 3.10
- **PyTorch** 2.5.*
- **Lightning** 2.x
- **Hydra** 1.3.2
