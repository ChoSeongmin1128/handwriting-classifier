# src/resnet_model.py
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, Flatten, Dense
from keras.models import Model

def residual_block(x, filters, strides=1):
    """ResNet의 기본 빌딩 블록인 잔차 블록(Residual Block)을 생성합니다."""
    shortcut = x
    
    y = Conv2D(filters, (3, 3), strides=strides, padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    # Grad-CAM을 위해 마지막 Conv Layer에 이름 지정
    y = Conv2D(filters, (3, 3), padding='same', name=f"res_conv_{filters}_{strides}")(y)
    y = BatchNormalization()(y)

    if strides != 1 or x.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=strides, padding='same')(x)
        shortcut = BatchNormalization()(shortcut)

    y = Add()([shortcut, y])
    y = Activation('relu')(y)
    return y

def create_resnet(input_shape=(28, 28, 1), num_classes=62):
    """EMNIST 데이터셋을 위한 간단한 ResNet 모델을 생성합니다."""
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x) 
    x = residual_block(x, filters=32)
    # 마지막 컨볼루션 레이어를 포함하는 블록을 쉽게 찾기 위해 이름 부여
    x = residual_block(x, filters=64, strides=2)
    
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name="resnet")
    return model