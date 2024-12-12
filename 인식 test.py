import cv2
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO('yolov8n.pt')  # 'yolov8n.pt', 'yolov8s.pt' 등 사용 가능

# 클래스 필터 (사람만 탐지)
CLASS_FILTER = ['person']

def process_frame(frame):
    # YOLOv8로 객체 탐지
    results = model(frame, verbose=False)

    # 결과 시각화
    for result in results[0].boxes:
        cls = result.cls.numpy()[0]
        if model.names[int(cls)] in CLASS_FILTER:
            x1, y1, x2, y2 = map(int, result.xyxy.numpy()[0])
            conf = result.conf.numpy()[0]
            label = f"{model.names[int(cls)]} {conf:.2f}"
            # 경계 상자 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def main():
    cap = cv2.VideoCapture(0)  # 웹캠 열기 (기본 장치)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 처리
        frame = process_frame(frame)

        # 결과 표시
        cv2.imshow("YOLOv8 Real-Time Detection", frame)

        # 'q'를 눌러 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
