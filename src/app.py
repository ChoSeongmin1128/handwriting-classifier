# src/app.py
import os
import sys
import time
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import io
import base64

# 현재 파일의 상위 디렉토리(handwriting-classifier)를 기준으로 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # handwriting-classifier 폴더

# Flask 앱 생성 시 templates와 static 폴더 경로 지정
app = Flask(__name__, 
            template_folder=os.path.join(project_root, 'templates'),
            static_folder=os.path.join(project_root, 'static'))

# EMNIST byclass 클래스 매핑 (62개 클래스)
def get_class_mapping():
    """EMNIST byclass 데이터셋의 클래스 인덱스를 실제 문자로 매핑합니다."""
    classes = []
    # 숫자 0-9
    for i in range(10):
        classes.append(str(i))
    # 대문자 A-Z
    for i in range(26):
        classes.append(chr(ord('A') + i))
    # 소문자 a-z
    for i in range(26):
        classes.append(chr(ord('a') + i))
    return classes

CLASS_NAMES = get_class_mapping()

# 모델 로딩
models = {}

def load_models():
    """학습된 모델들을 메모리에 로드합니다."""
    models_dir = os.path.join(project_root, 'models')
    
    model_paths = {
        'cnn': os.path.join(models_dir, 'baseline_cnn_emnist.keras'),
        'resnet': os.path.join(models_dir, 'resnet_emnist.keras')
    }
    
    print(f"📁 모델 디렉토리: {models_dir}")
    
    for model_name, path in model_paths.items():
        if os.path.exists(path):
            try:
                models[model_name] = tf.keras.models.load_model(path)
                print(f"✅ {model_name} 모델 로드 완료: {path}")
            except Exception as e:
                print(f"❌ {model_name} 모델 로드 실패: {e}")
        else:
            print(f"❌ {model_name} 모델 파일을 찾을 수 없습니다: {path}")

def save_debug_image(image_array, filename):
    """디버깅을 위해 전처리된 이미지를 저장합니다."""
    try:
        # 이미지 배열을 PIL 이미지로 변환
        debug_image = (image_array.reshape(28, 28) * 255).astype(np.uint8)
        debug_pil = Image.fromarray(debug_image, mode='L')
        
        # 크기를 키워서 확인하기 쉽게
        debug_pil = debug_pil.resize((280, 280), Image.NEAREST)
        
        debug_dir = os.path.join(project_root, 'debug_images')
        os.makedirs(debug_dir, exist_ok=True)
        
        debug_pil.save(os.path.join(debug_dir, filename))
        print(f"🔍 디버그 이미지 저장: {filename}")
    except Exception as e:
        print(f"❌ 디버그 이미지 저장 실패: {e}")

def preprocess_image(image_data):
    """개선된 이미지 전처리 - EMNIST 스타일에 최적화"""
    try:
        # base64 디코딩
        image_data = image_data.split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        
        # 그레이스케일 변환
        if image.mode != 'L':
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            image = image.convert('L')
        
        # numpy 배열로 변환
        img_array = np.array(image)
        
        # 배경을 흰색(255), 글씨를 검은색(0)으로 정규화
        img_array = 255 - img_array  # 반전
        
        # 글씨 영역 찾기 (더 관대한 임계값)
        binary = img_array > 30  # 임계값 낮춤
        
        if np.any(binary):
            # 글씨가 있는 영역의 경계 찾기
            rows = np.any(binary, axis=1)
            cols = np.any(binary, axis=0)
            if np.any(rows) and np.any(cols):
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]
                
                # 여백 추가 (비율 기반)
                height, width = img_array.shape
                margin_y = max(5, int((y_max - y_min) * 0.1))
                margin_x = max(5, int((x_max - x_min) * 0.1))
                
                y_min = max(0, y_min - margin_y)
                y_max = min(height, y_max + margin_y)
                x_min = max(0, x_min - margin_x)
                x_max = min(width, x_max + margin_x)
                
                # 크롭
                cropped = img_array[y_min:y_max, x_min:x_max]
            else:
                cropped = img_array
        else:
            cropped = img_array
        
        # PIL 이미지로 변환하여 리사이즈
        cropped_pil = Image.fromarray(cropped.astype(np.uint8), mode='L')
        
        # 정사각형으로 만들기
        w, h = cropped_pil.size
        size = max(w, h)
        
        # 정사각형 캔버스에 중앙 배치
        square = Image.new('L', (size, size), 0)  # 검은 배경
        offset = ((size - w) // 2, (size - h) // 2)
        square.paste(cropped_pil, offset)
        
        # 20x20으로 리사이즈 후 28x28에 중앙 배치
        resized = square.resize((20, 20), Image.LANCZOS)
        final = Image.new('L', (28, 28), 0)
        final.paste(resized, (4, 4))
        
        # 최종 배열 변환
        final_array = np.array(final, dtype=np.float32) / 255.0
        
        # 모델 입력 형태로 reshape
        final_array = final_array.reshape(1, 28, 28, 1)
        
        return final_array
        
    except Exception as e:
        print(f"❌ 이미지 전처리 오류: {e}")
        return None

@app.route('/')
def index():
    """메인 페이지를 렌더링합니다."""
    available_models = list(models.keys())
    return render_template('index.html', models=available_models)

@app.route('/predict', methods=['POST'])
def predict():
    """손글씨 이미지를 받아서 예측 결과를 반환합니다."""
    try:
        data = request.get_json()
        image_data = data.get('image')
        model_name = data.get('model', 'cnn')
        
        if not image_data:
            return jsonify({'error': '이미지 데이터가 없습니다.'}), 400
            
        if model_name not in models:
            return jsonify({'error': f'{model_name} 모델을 찾을 수 없습니다.'}), 400
        
        # 이미지 전처리
        processed_image = preprocess_image(image_data)
        if processed_image is None:
            return jsonify({'error': '이미지 전처리에 실패했습니다.'}), 400
        
        # 디버깅: 전처리된 이미지 저장
        timestamp = int(time.time())
        save_debug_image(processed_image[0], f"debug_{model_name}_{timestamp}.png")
        
        # 예측 수행
        model = models[model_name]
        predictions = model.predict(processed_image, verbose=0)
        
        # 상위 5개 예측 결과
        top5_indices = np.argsort(predictions[0])[-5:][::-1]
        
        results = []
        for idx in top5_indices:
            results.append({
                'character': CLASS_NAMES[idx],
                'confidence': float(predictions[0][idx])
            })
        
        print(f"🔮 예측 결과 ({model_name}): {results[0]['character']} ({results[0]['confidence']:.3f})")
        
        return jsonify({
            'success': True,
            'model_used': model_name,
            'predictions': results,
            'debug_image_saved': f"debug_{model_name}_{timestamp}.png"
        })
        
    except Exception as e:
        print(f"❌ 예측 오류: {e}")
        return jsonify({'error': f'예측 중 오류가 발생했습니다: {str(e)}'}), 500

@app.route('/test_model/<model_name>')
def test_model(model_name):
    """EMNIST 테스트 데이터로 모델 성능을 확인합니다."""
    if model_name not in models:
        return jsonify({'error': f'{model_name} 모델을 찾을 수 없습니다.'}), 400
    
    try:
        import tensorflow_datasets as tfds
        
        # EMNIST 테스트 데이터 로드
        def preprocess_emnist(image, label):
            image = tf.image.rot90(image, k=3)
            image = tf.image.flip_left_right(image)
            image = tf.cast(image, tf.float32) / 255.0
            return image, label
        
        test_ds = tfds.load('emnist/byclass', split='test[:1000]', as_supervised=True)
        test_ds = test_ds.map(preprocess_emnist).batch(32)
        
        # 모델 평가
        model = models[model_name]
        loss, accuracy = model.evaluate(test_ds, verbose=0)
        
        return jsonify({
            'model': model_name,
            'test_accuracy': float(accuracy),
            'test_loss': float(loss),
            'status': 'Model is working correctly' if accuracy > 0.7 else 'Model performance is low'
        })
        
    except Exception as e:
        return jsonify({'error': f'모델 테스트 실패: {str(e)}'}), 500

@app.route('/health')
def health():
    """서버 상태를 확인합니다."""
    return jsonify({
        'status': 'healthy',
        'loaded_models': list(models.keys()),
        'total_classes': len(CLASS_NAMES),
        'project_root': project_root,
        'current_dir': current_dir
    })

if __name__ == '__main__':
    print("🚀 손글씨 인식 웹 애플리케이션을 시작합니다...")
    print(f"📁 프로젝트 루트: {project_root}")
    print(f"📁 현재 디렉토리: {current_dir}")
    
    load_models()
    
    if not models:
        print("❌ 로드된 모델이 없습니다. 먼저 모델을 학습해주세요.")
        print("💡 다음 명령어로 모델을 학습하세요:")
        print("   python train.py --model cnn")
        print("   python train.py --model resnet")
        exit(1)
    
    print(f"✅ {len(models)}개의 모델이 로드되었습니다: {list(models.keys())}")
    print("🌐 서버 주소: http://localhost:5000")
    
    # WSL에서 외부 접근을 위해 host='0.0.0.0' 설정
    app.run(host='0.0.0.0', port=5000, debug=True)
