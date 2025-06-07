// static/script.js
class HandwritingApp {
    constructor() {
        this.canvas = document.getElementById('drawing-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.isDrawing = false;
        this.lastX = 0;
        this.lastY = 0;
        
        this.initCanvas();
        this.bindEvents();
        this.setupResponsiveCanvas();
    }

    initCanvas() {
        // 캔버스 설정 - EMNIST 스타일에 맞춤
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
        this.ctx.strokeStyle = '#000';
        this.ctx.lineWidth = 12;  // 두꺼운 선
        this.ctx.fillStyle = '#fff';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // 부드러운 그리기를 위한 설정
        this.ctx.globalCompositeOperation = 'source-over';
    }

    setupResponsiveCanvas() {
        // 모바일에서 캔버스 크기 조정
        const updateCanvasSize = () => {
            const container = this.canvas.parentElement;
            const maxSize = Math.min(container.clientWidth - 20, 280);
            
            if (window.innerWidth <= 480) {
                this.canvas.style.width = '250px';
                this.canvas.style.height = '250px';
            } else {
                this.canvas.style.width = '280px';
                this.canvas.style.height = '280px';
            }
        };

        updateCanvasSize();
        window.addEventListener('resize', updateCanvasSize);
    }

    bindEvents() {
        // 마우스 이벤트
        this.canvas.addEventListener('mousedown', (e) => this.startDrawing(e));
        this.canvas.addEventListener('mousemove', (e) => this.draw(e));
        this.canvas.addEventListener('mouseup', () => this.stopDrawing());
        this.canvas.addEventListener('mouseout', () => this.stopDrawing());

        // 터치 이벤트 (모바일 지원)
        this.canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent('mousedown', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            this.canvas.dispatchEvent(mouseEvent);
        });

        this.canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent('mousemove', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            this.canvas.dispatchEvent(mouseEvent);
        });

        this.canvas.addEventListener('touchend', (e) => {
            e.preventDefault();
            const mouseEvent = new MouseEvent('mouseup', {});
            this.canvas.dispatchEvent(mouseEvent);
        });

        // 버튼 이벤트
        document.getElementById('clear-btn').addEventListener('click', () => this.clearCanvas());
        document.getElementById('predict-btn').addEventListener('click', () => this.predict());

        // 모델 선택 변경 시 결과 숨기기
        document.getElementById('model-select').addEventListener('change', () => this.hideResults());
    }

    getMousePos(e) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        };
    }

    startDrawing(e) {
        this.isDrawing = true;
        const pos = this.getMousePos(e);
        [this.lastX, this.lastY] = [pos.x, pos.y];
        
        // 점 찍기 (클릭만 해도 점이 보이도록)
        this.ctx.beginPath();
        this.ctx.arc(pos.x, pos.y, this.ctx.lineWidth / 2, 0, Math.PI * 2);
        this.ctx.fill();
    }

    draw(e) {
        if (!this.isDrawing) return;
        
        const pos = this.getMousePos(e);
        
        // 선 그리기
        this.ctx.globalCompositeOperation = 'source-over';
        this.ctx.beginPath();
        this.ctx.moveTo(this.lastX, this.lastY);
        this.ctx.lineTo(pos.x, pos.y);
        this.ctx.stroke();
        
        // 부드러운 연결을 위해 원 그리기
        this.ctx.beginPath();
        this.ctx.arc(pos.x, pos.y, this.ctx.lineWidth / 2, 0, Math.PI * 2);
        this.ctx.fill();
        
        [this.lastX, this.lastY] = [pos.x, pos.y];
    }

    stopDrawing() {
        this.isDrawing = false;
    }

    clearCanvas() {
        this.ctx.fillStyle = '#fff';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.hideResults();
    }

    async predict() {
        const selectedModel = document.getElementById('model-select').value;
        const imageData = this.canvas.toDataURL('image/png');
        
        // 로딩 상태 표시
        const predictBtn = document.getElementById('predict-btn');
        const originalText = predictBtn.textContent;
        predictBtn.textContent = '🔍 예측 중...';
        predictBtn.disabled = true;

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: imageData,
                    model: selectedModel
                })
            });

            const result = await response.json();
            
            if (result.success) {
                this.displayResults(result);
            } else {
                this.showError('예측 실패: ' + result.error);
            }
        } catch (error) {
            console.error('예측 오류:', error);
            this.showError('서버와 통신 중 오류가 발생했습니다.');
        } finally {
            // 로딩 상태 해제
            predictBtn.textContent = originalText;
            predictBtn.disabled = false;
        }
    }

    displayResults(result) {
        const resultsDiv = document.getElementById('results');
        const predictionsDiv = document.getElementById('predictions');
        const modelUsedDiv = document.getElementById('model-used');
        
        // 사용된 모델 표시
        modelUsedDiv.textContent = `${result.model_used.toUpperCase()} 모델 사용`;
        
        predictionsDiv.innerHTML = '';
        
        const rankEmojis = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣'];
        
        result.predictions.forEach((prediction, index) => {
            const item = document.createElement('div');
            item.className = 'prediction-item';
            
            const confidence = (prediction.confidence * 100).toFixed(1);
            
            item.innerHTML = `
                <div style="display: flex; align-items: center;">
                    <span class="prediction-rank">${rankEmojis[index]}</span>
                    <span class="character">${prediction.character}</span>
                </div>
                <span class="confidence">${confidence}%</span>
            `;
            
            // 첫 번째 결과에 애니메이션 추가
            if (index === 0) {
                item.style.transform = 'scale(0.9)';
                setTimeout(() => {
                    item.style.transform = 'scale(1)';
                    item.style.transition = 'transform 0.3s ease';
                }, 100);
            }
            
            predictionsDiv.appendChild(item);
        });
        
        resultsDiv.classList.remove('hidden');
        
        // 결과 섹션으로 부드럽게 스크롤
        resultsDiv.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'nearest' 
        });
    }

    showError(message) {
        // 에러 메시지를 모달이나 알림으로 표시
        const errorDiv = document.createElement('div');
        errorDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #dc3545;
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 1000;
            max-width: 300px;
        `;
        errorDiv.textContent = message;
        
        document.body.appendChild(errorDiv);
        
        // 3초 후 자동 제거
        setTimeout(() => {
            errorDiv.remove();
        }, 3000);
    }

    hideResults() {
        document.getElementById('results').classList.add('hidden');
    }
}

// 앱 초기화
document.addEventListener('DOMContentLoaded', () => {
    new HandwritingApp();
    
    // 페이지 로드 완료 알림
    console.log('🎉 손글씨 인식 앱이 준비되었습니다!');
});

// 서비스 워커 등록 (PWA 지원을 위한 기본 설정)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        // 추후 PWA 구현 시 사용
    });
}
