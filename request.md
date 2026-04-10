## 목표

MNIST 손글씨 숫자 분류의 검증 정확도(val/acc)를 최대한 높여주세요.

현재 베이스라인은 SimpleDenseNet(FC 4층: 784→64→128→64→10)이며, 아키텍처 변경·하이퍼파라미터 튜닝·데이터 증강 등 자유롭게 시도해도 됩니다.

## 데이터

- 데이터셋: MNIST (28×28 그레이스케일, 10클래스)
- 경로: `data/` (자동 다운로드)
- 분할: train 55,000 / val 5,000 / test 10,000
- 전처리: ToTensor + Normalize(mean=0.1307, std=0.3081)

## 학습 실행 명령어

```bash
uv run python src/train.py experiment=mnist
```

## 메트릭

- val/acc (maximize)

학습 완료 후 stdout에 `val/acc`와 `test/acc`가 출력됩니다.

## 제약 조건

- 수정 가능 파일:
  - `src/models/components/simple_dense_net.py` (모델 아키텍처)
  - `src/models/mnist_module.py` (LightningModule)
  - `src/data/mnist_datamodule.py` (데이터 모듈)
  - `configs/model/mnist.yaml` (모델 하이퍼파라미터)
  - `configs/data/mnist.yaml` (데이터 설정)
  - `configs/experiment/mnist.yaml` (최종 실험 설정, 오버라이딩.)
- 에폭 수는 20 이하
- 외부 사전학습 가중치 사용 금지
- 데이터셋 변경 금지 (MNIST 유지)
