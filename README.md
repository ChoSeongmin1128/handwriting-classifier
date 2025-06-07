### README.md 파일 보충

제공해주신 `README.md` 파일에 현재 프로젝트 상태와 기능을 더 잘 반영하고, 사용자에게 더 유용한 정보를 제공할 수 있도록 다음과 같이 보충 내용을 제안합니다.

---

```markdown
# 손글씨 인식 프로젝트

EMNIST 데이터셋을 사용한 딥러닝 기반 손글씨 인식 웹 애플리케이션

## 프로젝트 개요

이 프로젝트는 CNN과 ResNet 모델을 사용하여 손글씨 문자(숫자, 영문자)를 인식하는 시스템입니다. 웹 인터페이스를 통해 실시간으로 손글씨를 그리고 예측 결과를 확인할 수 있습니다.

### 주요 기능
- **62개 클래스 인식** (숫자 0-9, 대문자 A-Z, 소문자 a-z)
- **두 가지 딥러닝 모델 비교** (Baseline CNN 및 Custom ResNet)
    - 정확도, 추론 시간, 파라미터 수 등 다양한 지표 비교
    - 클래스별 정확도 히트맵 및 혼동 행렬 차이 시각화
    - 상세 비교 보고서 (`.txt` 파일) 생성
- **실시간 웹 인터페이스** (Gradio 기반)
- **예측 결과 상위 N개 표시** (기본 상위 5개)
- **디버깅용 이미지 전처리 단계 저장** (옵션)

## 사용 기술
- **딥러닝 프레임워크**: TensorFlow 2.19.0 (GPU 지원 포함), Keras
- **데이터셋**: EMNIST byclass (TensorFlow Datasets 활용)
- **웹 인터페이스**: Gradio
- **데이터 처리**: NumPy, Pandas, scikit-learn
- **시각화**: Matplotlib, Seaborn (한글 폰트 지원)
- **개발 환경**: WSL2 (Ubuntu), Python 3.10.18
- **웹 서버**: Flask (기존 app.py가 Flask였다면) 또는 Gradio 내장 서버 (새 app.py에 따라)

## 설치 및 실행

이 프로젝트는 WSL2 (Windows Subsystem for Linux 2) 환경에서 최적화되어 있습니다. GPU 가속을 위해 NVIDIA GPU와 최신 드라이버가 필요합니다.

### 1. WSL2 환경 준비

WSL2 Ubuntu 배포판이 설치되어 있고, NVIDIA GPU 드라이버가 Windows에 설치되어 있는지 확인합니다.

### 2. Python 가상 환경 설정

프로젝트를 위한 독립적인 Python 가상 환경을 구축합니다. TensorFlow와 PyTorch를 독립적으로 사용하기 위해 별도의 가상 환경을 권장합니다.

```bash
# 프로젝트 디렉토리로 이동
# 예시: cd ~/projects/tensorflow_project/handwriting-classifier/

# TensorFlow 전용 가상 환경 생성 및 활성화
python3.10 -m venv ../tensorflow_env
source ../tensorflow_env/bin/activate

# 필수 패키지 업그레이드 (선택 사항이지만 권장)
pip install --upgrade pip setuptools wheel

# TensorFlow 및 관련 라이브러리 설치
# requirements.txt 파일에 명시된 패키지를 설치합니다.
pip install -r requirements.txt
```

### 3. 모델 학습

모델을 학습시키고 `models/` 디렉토리에 저장합니다. 두 가지 모델을 모두 학습시키는 것을 권장합니다.

```bash
# CNN 모델 학습
python src/train.py --model cnn

# ResNet 모델 학습
python src/train.py --model resnet
```
*학습된 모델 파일은 `models/baseline_cnn_emnist.keras`와 `models/resnet_emnist.keras`로 저장됩니다.*

### 4. 모델 평가 및 비교 (선택 사항)

학습된 모델의 성능을 평가하고 시각화된 비교 보고서를 생성할 수 있습니다.

```bash
# 개별 모델 평가 (예: ResNet 모델)
# results/confusion_matrix.png 및 results/grad_cam_examples.png 생성
python src/evaluate.py --model models/resnet_emnist.keras --samples 2000

# 두 모델 비교
# results/ 디렉토리에 다양한 비교 차트 및 report.txt 생성
python src/compare_models.py --model1 models/baseline_cnn_emnist.keras --model2 models/resnet_emnist.keras --samples 5000 --output results
```
*`results/` 디렉토리는 자동으로 생성됩니다.*

### 5. 웹 애플리케이션 실행

Gradio 기반의 웹 인터페이스를 실행하여 실시간 손글씨 인식을 테스트합니다.

```bash
python src/app.py
```
브라우저에서 `http://localhost:7860` (또는 터미널에 표시되는 URL)에 접속합니다.

## 프로젝트 구조
```
handwriting-classifier/
├── src/
│   ├── app.py              # Gradio 웹 애플리케이션
│   ├── baseline_cnn.py     # Baseline CNN 모델 정의
│   ├── resnet_model.py     # ResNet 모델 정의
│   ├── train.py            # 모델 학습 스크립트
│   ├── evaluate.py         # 모델 평가 스크립트 (혼동 행렬, Grad-CAM)
│   └── compare_models.py   # 두 모델 비교 스크립트
├── models/                 # 학습된 모델 저장 (.keras 파일)
├── results/                # 평가 및 비교 결과 이미지, 보고서 저장
├── debug_images/           # (선택 사항) app.py 실행 시 전처리된 이미지 저장
├── templates/              # (Flask 사용 시) HTML 템플릿
├── static/                 # (Flask 사용 시) CSS, JS 등 정적 파일
└── README.md
└── requirements.txt        # Python 패키지 의존성 파일
```

## 참고사항

- **WSL2 환경**: 이 프로젝트는 WSL2 (Ubuntu) 환경에서 개발 및 테스트되었습니다. 