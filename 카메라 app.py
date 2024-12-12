from flask import Flask, render_template, Response, request
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# YOLOv8 모델 로드
model = YOLO("yolov8n.pt")

def generateframes():
    camera = cv2.VideoCapture(0)  # 웹캠 열기
    while True:
        success, frame = camera.read()
        if not success:
            break

        # YOLOv8 객체 탐지
        results = model(frame)
        annotatedframe = results[0].plot()  # 결과를 시각화

        # 프레임을 인코딩하여 전송
        ret, buffer = cv2.imencode('.jpg', annotatedframe)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')  # HTML 렌더링

@app.route('/video_feed')
def video_feed():
    # 여기에서 generateframes() 함수 호출
    return Response(generateframes(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
