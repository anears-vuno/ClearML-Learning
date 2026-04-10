data_path: data/MNIST
학습 스크립트: uv run src/train.py experiment=mnist
목표: val/acc 최대화 시키기

- 각 실험은 단일 GPU에서 실행
- 실험은 5분의 고정된 시간동안 실행
- 수정가능한 파일
    - configs/experiment/mnist.yaml
    - src/models 폴더 안 파일들
