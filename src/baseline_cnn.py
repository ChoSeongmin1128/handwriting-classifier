# src/baseline_cnn.py
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_baseline_cnn(input_shape=(28, 28, 1), num_classes=62):
    """비교 실험을 위한 간단한 기본 CNN 모델을 생성합니다."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', name="last_conv_layer"), # Grad-CAM을 위해 이름 지정
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ], name="baseline_cnn")
    return model