# src/app.py (디버깅용 전체 코드)
import gradio as gr
import numpy as np
import tensorflow as tf
import os
from PIL import Image

# 클래스 이름 정의 (0-9, A-Z, a-z)
CLASS_NAMES = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]

# 모델 로드 함수
def load_best_model():
    """사용 가능한 최고 성능 모델을 로드합니다."""
    model_paths = [
        'models/resnet_emnist.keras',
        'models/baseline_cnn_emnist.keras'
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                print(f"Loading model: {path}")
                return tf.keras.models.load_model(path)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
    
    return None

# 모델 로드
model = load_best_model()

def predict_character(image):
    """손글씨 이미지를 분류합니다."""
    if model is None:
        print("Error: 모델을 찾을 수 없습니다.]")
        return {}

    if image is None:
        print("Error: 이미지를 그려주세요.")
        return {}

    try:
        # === 디버깅 정보 추가 ===
        print("\n" + "="*50)
        print(f"🔍 입력 데이터 타입: {type(image)}")
        print(f"🔍 입력 데이터 구조: {image.keys() if isinstance(image, dict) else 'Not a dict'}")
        
        # dict 형태인 경우 처리
        if isinstance(image, dict):
            # 가능한 모든 키 확인
            print(f"🔍 사용 가능한 키들: {list(image.keys())}")
            
            # 각 키의 내용도 확인
            for key, value in image.items():
                if value is not None:
                    if isinstance(value, np.ndarray):
                        print(f"  - {key}: numpy array {value.shape}, dtype={value.dtype}")
                    elif isinstance(value, Image.Image):
                        print(f"  - {key}: PIL Image {value.size}, mode={value.mode}")
                    else:
                        print(f"  - {key}: {type(value)}")
                else:
                    print(f"  - {key}: None")
            
            # 가장 적절한 키 선택
            if 'composite' in image and image['composite'] is not None:
                image_data = image['composite']
                print("✅ 'composite' 키 사용")
            elif 'image' in image and image['image'] is not None:
                image_data = image['image']
                print("✅ 'image' 키 사용")
            else:
                # 첫 번째 None이 아닌 값 사용
                image_data = None
                for key, value in image.items():
                    if value is not None:
                        image_data = value
                        print(f"✅ '{key}' 키 사용")
                        break
                
                if image_data is None:
                    print("❌ 사용 가능한 이미지 데이터가 없습니다.")
                    return {}
        else:
            image_data = image
            print("✅ 직접 이미지 데이터 사용")
        
        # 이미지 데이터 정보 출력
        if isinstance(image_data, np.ndarray):
            print(f"🔍 이미지 배열 shape: {image_data.shape}")
            print(f"🔍 이미지 배열 dtype: {image_data.dtype}")
            print(f"🔍 이미지 값 범위: {image_data.min()} ~ {image_data.max()}")
        elif isinstance(image_data, Image.Image):
            print(f"🔍 PIL 이미지 크기: {image_data.size}")
            print(f"🔍 PIL 이미지 모드: {image_data.mode}")
        
        # numpy 배열로 변환
        if isinstance(image_data, Image.Image):
            image_array = np.array(image_data)
            print("🔄 PIL Image를 numpy로 변환")
        elif isinstance(image_data, np.ndarray):
            image_array = image_data
            print("🔄 이미 numpy 배열")
        else:
            print(f"❌ 지원하지 않는 이미지 형태: {type(image_data)}")
            return {}
        
        print(f"🔍 변환 후 shape: {image_array.shape}")
        print(f"🔍 변환 후 dtype: {image_array.dtype}")
        
        # 그레이스케일 변환
        if len(image_array.shape) == 3:
            if image_array.shape[2] == 4:  # RGBA
                print("🔄 RGBA에서 Alpha 채널 추출")
                alpha_channel = image_array[:, :, 3]
                print(f"🔍 Alpha 채널 값 범위: {alpha_channel.min()} ~ {alpha_channel.max()}")
                image_array = alpha_channel
            elif image_array.shape[2] == 3:  # RGB
                print("🔄 RGB를 그레이스케일로 변환")
                image_array = np.dot(image_array, [0.299, 0.587, 0.114])
            else:
                print(f"🔍 예상하지 못한 채널 수: {image_array.shape[2]}")
        elif len(image_array.shape) == 2:
            print("✅ 이미 그레이스케일")
        else:
            print(f"❌ 예상하지 못한 이미지 차원: {image_array.shape}")
        
        print(f"🔍 그레이스케일 변환 후 shape: {image_array.shape}")
        print(f"🔍 그레이스케일 값 범위: {image_array.min()} ~ {image_array.max()}")
        
        # 빈 이미지 검사
        unique_values = np.unique(image_array)
        print(f"🔍 유니크 값 개수: {len(unique_values)}")
        if len(unique_values) <= 10:
            print(f"🔍 유니크 값들: {unique_values}")
        else:
            print(f"🔍 유니크 값들 (일부): {unique_values[:10]}...")
        
        if len(unique_values) <= 2:
            print("⚠️ 경고: 이미지가 비어있거나 단순할 수 있습니다.")
            if np.all(image_array == 0):
                print("❌ 완전히 검은 이미지")
                return {"빈 이미지": 1.0}
            elif np.all(image_array == 255):
                print("❌ 완전히 흰 이미지")
                return {"빈 이미지": 1.0}
        
        # 28x28로 리사이즈
        original_shape = image_array.shape
        if image_array.shape != (28, 28):
            print(f"🔄 {original_shape}에서 (28, 28)로 리사이즈")
            image_pil = Image.fromarray(image_array.astype(np.uint8))
            image_pil = image_pil.resize((28, 28), Image.LANCZOS)
            image_array = np.array(image_pil)
        else:
            print("✅ 이미 28x28 크기")
        
        print(f"🔍 리사이즈 후 값 범위: {image_array.min()} ~ {image_array.max()}")
        
        # 원본 이미지 정보 저장 (디버깅용)
        original_for_debug = image_array.copy()
        
        # 색상 반전 테스트 (주석 해제해서 테스트)
        print("🔄 색상 반전 적용 중...")
        image_array = 255 - image_array
        print(f"🔍 색상 반전 후 값 범위: {image_array.min()} ~ {image_array.max()}")
        
        # EMNIST 변환 테스트 (주석 해제해서 테스트)
        print("🔄 EMNIST 형태 변환 적용 중...")
        image_array = np.fliplr(np.rot90(image_array, k=1))
        print(f"🔍 EMNIST 변환 후 값 범위: {image_array.min()} ~ {image_array.max()}")
        
        # 최종 전처리 상태 확인
        unique_final = np.unique(image_array)
        print(f"🔍 최종 유니크 값 개수: {len(unique_final)}")
        
        # 정규화
        img_normalized = image_array.astype(np.float32) / 255.0
        img_batch = img_normalized.reshape(1, 28, 28, 1)
        
        print(f"🔍 정규화 후 값 범위: {img_normalized.min():.4f} ~ {img_normalized.max():.4f}")
        print(f"🔍 배치 shape: {img_batch.shape}")
        
        # 예측 수행
        print("🔄 모델 예측 중...")
        prediction = model.predict(img_batch, verbose=0).flatten()
        
        # 예측 결과 분석
        print(f"🔍 예측 벡터 shape: {prediction.shape}")
        print(f"🔍 예측 값 범위: {prediction.min():.4f} ~ {prediction.max():.4f}")
        print(f"🔍 예측 값 합계: {prediction.sum():.4f}")
        
        # 상위 10개 결과 확인 (디버깅용)
        top_10_indices = np.argsort(prediction)[-10:][::-1]
        print("🔍 상위 10개 예측:")
        for i, idx in enumerate(top_10_indices):
            print(f"  {i+1}. {CLASS_NAMES[idx]}: {prediction[idx]:.4f}")
        
        # 상위 5개 결과 반환
        top_indices = np.argsort(prediction)[-5:][::-1]
        result = {CLASS_NAMES[i]: float(prediction[i]) for i in top_indices}
        
        print(f"🎯 최종 결과: {result}")
        print("="*50 + "\n")
        
        return result

    except Exception as e:
        print(f"❌ 예측 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

# Gradio 인터페이스 생성
def create_interface():
    """Gradio 웹 인터페이스를 생성합니다."""
    iface = gr.Interface(
        fn=predict_character,
        inputs=gr.Sketchpad(
            canvas_size=(280, 280),  # 캔버스 크기
        ),
        outputs=gr.Label(num_top_classes=5),
        live=True,  # 실시간 예측
        title="🖋️ 손글씨 영문/숫자 인식기 (디버깅 모드)",
        description="""
        **디버깅 모드에서 실행 중입니다**
        
        **사용법:**
        1. 왼쪽 캔버스에 숫자(0-9) 또는 영문자(A-Z, a-z)를 그려보세요
        2. 콘솔 창에서 상세한 처리 과정을 확인할 수 있습니다
        3. 오른쪽에 예측 결과가 표시됩니다
        
        **팁:** 
        - 캔버스 중앙에 크고 명확하게 글자를 그려주세요
        - 콘솔에서 이미지 처리 과정을 자세히 관찰하세요
        """,
        theme=gr.themes.Soft()
    )
    return iface

if __name__ == "__main__":
    if model is None:
        print("❌ 모델을 찾을 수 없습니다!")
        print("다음 명령어로 먼저 모델을 학습하세요:")
        print("  python src/train.py --model resnet")
        print("  또는")
        print("  python src/train.py --model cnn")
    else:
        print("✅ 모델 로드 완료!")
        print("🔍 디버깅 모드로 Gradio 서버를 시작합니다...")
        print("📋 콘솔에서 상세한 처리 과정을 확인하세요!")
        
        iface = create_interface()
        iface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )
