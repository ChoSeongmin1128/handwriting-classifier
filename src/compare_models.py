# src/compare_models.py
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras.models import load_model
import pandas as pd
import time
import os

# 한글 폰트 설정 (조성민님 선호도에 맞춰)
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

def preprocess(image, label):
    """EMNIST 데이터셋을 올바른 방향으로 변환하고 정규화합니다."""
    image = tf.image.rot90(image, k=3)
    image = tf.image.flip_left_right(image)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def load_test_data(num_samples=2000):
    """테스트 데이터를 로드하고 전처리합니다."""
    print(f"테스트 데이터 로딩 중... (샘플 수: {num_samples})")
    test_ds = tfds.load('emnist/byclass', split='test', as_supervised=True).map(preprocess)
    
    test_sample = test_ds.take(num_samples)
    x_test_list, y_true_list = [], []
    
    for image, label in test_sample:
        x_test_list.append(image.numpy())
        y_true_list.append(label.numpy())
    
    return np.array(x_test_list), np.array(y_true_list)

def evaluate_model(model, x_test, y_true, model_name):
    """모델을 평가하고 성능 지표를 반환합니다."""
    print(f"\n{model_name} 모델 평가 중...")
    
    # 추론 시간 측정
    start_time = time.time()
    y_pred_probs = model.predict(x_test, batch_size=32, verbose=0)
    end_time = time.time()
    
    inference_time = end_time - start_time
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # 성능 지표 계산
    accuracy = accuracy_score(y_true, y_pred)
    
    # 클래스별 정확도 계산
    class_names = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'inference_time': inference_time,
        'avg_inference_per_sample': inference_time / len(x_test),
        'y_pred': y_pred,
        'y_pred_probs': y_pred_probs,
        'num_parameters': model.count_params()
    }
    
    print(f"{model_name} 결과:")
    print(f"  정확도: {accuracy:.4f}")
    print(f"  총 추론 시간: {inference_time:.2f}초")
    print(f"  샘플당 평균 추론 시간: {inference_time/len(x_test)*1000:.2f}ms")
    print(f"  모델 파라미터 수: {model.count_params():,}")
    
    return results

def create_comparison_plots(results_list, y_true, save_dir='results'):
    """두 모델을 비교하는 다양한 시각화를 생성합니다."""
    os.makedirs(save_dir, exist_ok=True)
    class_names = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]
    
    # 1. 성능 지표 비교 바차트
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    model_names = [r['model_name'] for r in results_list]
    accuracies = [r['accuracy'] for r in results_list]
    inference_times = [r['avg_inference_per_sample']*1000 for r in results_list]  # ms 단위
    param_counts = [r['num_parameters']/1000 for r in results_list]  # K 단위
    
    # 정확도 비교
    bars1 = axes[0, 0].bar(model_names, accuracies, color=['skyblue', 'lightcoral'])
    axes[0, 0].set_title('모델별 정확도 비교', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('정확도')
    axes[0, 0].set_ylim(0, 1)
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{accuracies[i]:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 추론 시간 비교
    bars2 = axes[0, 1].bar(model_names, inference_times, color=['lightgreen', 'orange'])
    axes[0, 1].set_title('모델별 샘플당 추론 시간 비교', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('추론 시간 (ms)')
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{inference_times[i]:.2f}ms', ha='center', va='bottom', fontweight='bold')
    
    # 파라미터 수 비교
    bars3 = axes[1, 0].bar(model_names, param_counts, color=['plum', 'wheat'])
    axes[1, 0].set_title('모델별 파라미터 수 비교', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('파라미터 수 (K)')
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                       f'{param_counts[i]:.1f}K', ha='center', va='bottom', fontweight='bold')
    
    # 효율성 지표 (정확도/파라미터수)
    efficiency = [acc/param for acc, param in zip(accuracies, param_counts)]
    bars4 = axes[1, 1].bar(model_names, efficiency, color=['lightsteelblue', 'lightsalmon'])
    axes[1, 1].set_title('효율성 지표 (정확도/파라미터수)', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('효율성')
    for i, bar in enumerate(bars4):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                       f'{efficiency[i]:.6f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/model_comparison_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 클래스별 정확도 히트맵 비교
    fig, axes = plt.subplots(1, len(results_list), figsize=(20, 8))
    
    for i, result in enumerate(results_list):
        # 클래스별 정확도 계산
        class_accuracy = []
        for class_idx in range(62):
            mask = (y_true == class_idx)
            if np.sum(mask) > 0:
                class_acc = np.mean(result['y_pred'][mask] == y_true[mask])
                class_accuracy.append(class_acc)
            else:
                class_accuracy.append(0)
        
        # 10x7 그리드로 재배열 (숫자 10개 + 대문자 26개 + 소문자 26개)
        # EMNIST ByClass는 총 62개 클래스 (0-9, A-Z, a-z)
        # 7행 9열 (63칸) 그리드에 맞추기 위해 1개의 패딩 추가
        # 아니면 8행 8열 (64칸) 그리드에 2개의 패딩을 넣을 수도 있습니다.
        accuracy_grid = np.array(class_accuracy + [0]*(63 - len(class_accuracy))).reshape(7, 9)
        # 또는
        # accuracy_grid = np.array(class_accuracy + [0]*(64 - len(class_accuracy))).reshape(8, 8)
                
        sns.heatmap(accuracy_grid, annot=True, fmt='.3f', cmap='RdYlBu_r',
                   ax=axes[i], cbar_kws={'shrink': 0.8})
        axes[i].set_title(f'{result["model_name"]} 클래스별 정확도', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('클래스 인덱스')
        axes[i].set_ylabel('그룹')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/class_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 혼동 행렬 차이 시각화
    if len(results_list) == 2:
        cm1 = confusion_matrix(y_true, results_list[0]['y_pred'])
        cm2 = confusion_matrix(y_true, results_list[1]['y_pred'])
        
        # 정규화된 혼동 행렬
        cm1_norm = cm1.astype('float') / cm1.sum(axis=1)[:, np.newaxis]
        cm2_norm = cm2.astype('float') / cm2.sum(axis=1)[:, np.newaxis]
        
        # 차이 계산
        cm_diff = cm2_norm - cm1_norm
        
        plt.figure(figsize=(20, 16))
        sns.heatmap(cm_diff, annot=False, cmap='RdBu_r', center=0,
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'shrink': 0.8})
        plt.title(f'{results_list[1]["model_name"]} - {results_list[0]["model_name"]} 혼동행렬 차이\n(양수: {results_list[1]["model_name"]}이 더 좋음)', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('예측 라벨')
        plt.ylabel('실제 라벨')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/confusion_matrix_difference.png', dpi=300, bbox_inches='tight')
        plt.close()

def create_summary_report(results_list, save_dir='results'):
    """모델 비교 요약 보고서를 생성합니다."""
    report_path = f'{save_dir}/comparison_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("         모델 성능 비교 보고서\n")
        f.write("=" * 60 + "\n\n")
        
        for i, result in enumerate(results_list, 1):
            f.write(f"{i}. {result['model_name']} 모델\n")
            f.write("-" * 40 + "\n")
            f.write(f"정확도: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)\n")
            f.write(f"총 추론 시간: {result['inference_time']:.2f}초\n")
            f.write(f"샘플당 평균 추론 시간: {result['avg_inference_per_sample']*1000:.2f}ms\n")
            f.write(f"모델 파라미터 수: {result['num_parameters']:,}개\n")
            f.write(f"효율성 지표: {result['accuracy']/(result['num_parameters']/1000):.6f}\n\n")
        
        if len(results_list) == 2:
            r1, r2 = results_list
            f.write("비교 분석\n")
            f.write("-" * 40 + "\n")
            acc_diff = r2['accuracy'] - r1['accuracy']
            time_diff = r2['avg_inference_per_sample'] - r1['avg_inference_per_sample']
            param_diff = r2['num_parameters'] - r1['num_parameters']
            
            f.write(f"정확도 차이: {acc_diff:+.4f} ({acc_diff*100:+.2f}%p)\n")
            f.write(f"추론 시간 차이: {time_diff*1000:+.2f}ms\n")
            f.write(f"파라미터 수 차이: {param_diff:+,}개\n")
            
            if acc_diff > 0:
                f.write(f"\n✅ {r2['model_name']} 모델이 정확도가 더 높습니다.\n")
            else:
                f.write(f"\n✅ {r1['model_name']} 모델이 정확도가 더 높습니다.\n")
            
            if time_diff < 0:
                f.write(f"✅ {r2['model_name']} 모델이 추론 속도가 더 빠릅니다.\n")
            else:
                f.write(f"✅ {r1['model_name']} 모델이 추론 속도가 더 빠릅니다.\n")
    
    print(f"\n📊 상세 보고서가 {report_path}에 저장되었습니다.")

def main():
    parser = argparse.ArgumentParser(description='두 모델의 성능을 비교합니다.')
    parser.add_argument('--model1', type=str, required=True, help='첫 번째 모델 경로')
    parser.add_argument('--model2', type=str, required=True, help='두 번째 모델 경로')
    parser.add_argument('--samples', type=int, default=2000, help='평가할 샘플 수 (기본값: 2000)')
    parser.add_argument('--output', type=str, default='results', help='결과 저장 디렉토리')
    args = parser.parse_args()
    
    # 결과 디렉토리 생성
    os.makedirs(args.output, exist_ok=True)
    
    # 데이터 로드
    x_test, y_true = load_test_data(args.samples)
    
    # 모델들 로드 및 평가
    results_list = []
    
    for model_path in [args.model1, args.model2]:
        if not os.path.exists(model_path):
            print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
            return
        
        print(f"\n모델 로딩 중: {model_path}")
        model = load_model(model_path)
        model_name = os.path.basename(model_path).replace('.keras', '').replace('_emnist', '')
        
        result = evaluate_model(model, x_test, y_true, model_name)
        results_list.append(result)
    
    # 비교 시각화 및 보고서 생성
    print(f"\n📈 비교 시각화 생성 중...")
    create_comparison_plots(results_list, y_true, args.output)
    create_summary_report(results_list, args.output)
    
    print(f"\n🎉 모델 비교 완료!")
    print(f"결과 파일들이 '{args.output}' 디렉토리에 저장되었습니다:")
    print(f"  - model_comparison_metrics.png (성능 지표 비교)")
    print(f"  - class_accuracy_comparison.png (클래스별 정확도)")
    print(f"  - confusion_matrix_difference.png (혼동행렬 차이)")
    print(f"  - comparison_report.txt (상세 보고서)")

if __name__ == "__main__":
    main()
