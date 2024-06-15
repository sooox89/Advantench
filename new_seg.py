from collections import defaultdict
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# 색상 정의
CLASS_COLORS = {
    'container': (0, 255, 0),  # 녹색
    'corner-casting': (255, 0, 0)  # 빨강
}

track_history = defaultdict(lambda: [])
model = YOLO("best_seg.pt")

# 비디오 경로 설정
video_path = "/Users/sooox89/Desktop/advantench/video/IMG_7526.mp4"

cap = cv2.VideoCapture(video_path)
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

out = cv2.VideoWriter("instance-segmentation-object-tracking.avi", cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))

def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def is_containers_aligned(container_castings, threshold=10):
    for i in range(len(container_castings) - 1):
        for j in range(i + 1, len(container_castings)):
            for k in range(4):
                if container_castings[i][k] is None or container_castings[j][k] is None:
                    print(f"Skipping alignment check for None values at container {i} and {j}, corner {k}")
                    continue  # None 값이 있는 경우 비교를 건너뜀
                if abs(container_castings[i][k][0] - container_castings[j][k][0]) > threshold:
                    return False
    return True


while True:
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    annotator = Annotator(im0, line_width=2)

    results = model.track(im0, persist=True)

    if results[0].boxes.id is not None and results[0].masks is not None:
        masks = results[0].masks.xy
        track_ids = results[0].boxes.id.int().cpu().tolist()
        boxes = results[0].boxes.xyxy.tolist()
        classes = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.tolist()
        names = model.names

        for mask, track_id, box, cls, conf in zip(masks, track_ids, boxes, classes, confidences):
            class_name = names[cls]
            mask_color = CLASS_COLORS.get(class_name, (255, 255, 255))  # 기본 색상을 흰색으로 설정
            annotator.seg_bbox(mask=mask, mask_color=mask_color)

            x1, y1, x2, y2 = box  # 좌표를 그대로 유지
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            # 중심 좌표 표시
            cv2.circle(im0, (int(x_center), int(y_center)), 5, (0, 255, 255), -1)

            # 중심 좌표 값 표시
            center_label = f'({x_center:.2f}, {y_center:.2f})'
            cv2.putText(im0, center_label, (int(x_center), int(y_center) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mask_color, 2)

            # 라벨 추가 (소수점 포함)
            label = f'Track ID: {track_id} {class_name} {conf:.2f} Center: ({x_center:.2f}, {y_center:.2f})'
            annotator.text((int(x1), int(y1)), label, txt_color=(255, 255, 255))

    im0 = annotator.result()
    out.write(im0)
    cv2.imshow("instance-segmentation-object-tracking", im0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

out.release()
cap.release()
cv2.destroyAllWindows()