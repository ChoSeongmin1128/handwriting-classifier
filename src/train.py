# src/train.py
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.callbacks import ModelCheckpoint
from baseline_cnn import create_baseline_cnn
from resnet_model import create_resnet

# ✅ 수정: 인자를 image, label 두 개로 받도록 변경
def preprocess(image, label):
    """EMNIST 데이터셋을 올바른 방향으로 변환하고 정규화합니다."""
    # EMNIST 원본은 회전 및 반전되어 있으므로 올바른 방향으로 재조정
    image = tf.image.rot90(image, k=3)
    image = tf.image.flip_left_right(image)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# 데이터셋 준비
train_ds = tfds.load('emnist/byclass', split='train', as_supervised=True, shuffle_files=True).map(preprocess).batch(128)
test_ds = tfds.load('emnist/byclass', split='test', as_supervised=True).map(preprocess).batch(128)

# 커맨드 라인 인자로 학습할 모델 선택
parser = argparse.ArgumentParser(description='Train CNN or ResNet model on EMNIST dataset.')
parser.add_argument('--model', type=str, required=True, choices=['cnn', 'resnet'], help='Model to train: cnn or resnet')
args = parser.parse_args()

# 선택된 모델 생성
if args.model == 'cnn':
    model = create_baseline_cnn()
else:
    model = create_resnet()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 모델 학습 및 최적 모델 저장
checkpoint = ModelCheckpoint(
    filepath=f'models/{model.name}_emnist.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

print(f"\nTraining {model.name} model...")
model.fit(
    train_ds,
    epochs=10,
    validation_data=test_ds,
    callbacks=[checkpoint]
)
print(f"\nFinished training {model.name} model.")