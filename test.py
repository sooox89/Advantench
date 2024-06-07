# 이 코드는 라이다 값으로 threshold 설정하여,
# detect된 컨테이너들 중 가장 상단에 위치한 컨테이너의 상단 모서리와 현재 적재하는 컨테이너의 코너캐스팅 정렬 확인

# 1. 감지된 컨테이너가 없을 때, 컨테이너 상단 코너캐스팅 좌표와 감지된 코너캐스팅 좌표를 비교하도록 로직 추가
# 2. 감지된 컨테이너가 있을 때와 없을 때 모두 처리 가능하도록 수정

# 1) 컨테이너와 코너 캐스팅 좌표 수집
# 2) 각 모서리에 대해 가장 가까운 코너 캐스팅 찾기
# 3) 컨테이너 상단 코너와 상단에 위치한 코너 캐스팅 비교
# --------------수정 사항---------------------------
# 3-1) 감지된 코너 캐스팅이 없는 경우 : 컨테이너의 중심 좌표를 이용해서 정렬 탐지


import cv2
import numpy as np
import json
import torch
from ultralytics import YOLO

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


def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# 장치 설정 (GPU 사용 가능 여부에 따라)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 학습된 모델 가져오기 -> best.pt
model = YOLO("best.pt")

# 비디오 경로 설정
video_path = ""
# cap = cv2.VideoCapture(video_path)
cap = cv2.VideoCapture(0)


# 비디오 저장 설정 (H.264 코덱 사용)
fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 코덱 설정
output_path = "output_video.mp4"
out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# JSON 파일을 쓰기 모드로 열기
results_data = []
frame_count = 0

# 비디오 프레임을 반복 처리합니다.
while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break
    # 비디오 해상도 가져오기
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count += 1

    results = model.track(frame, persist=True)

    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    confidences = results[0].boxes.conf.tolist()
    names = results[0].names

    detection_results = {
        'frame': frame_count,
        'detections': []
    }
    container_count = 0
    corner_casting_count = 0

    for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
        # 여기서 confidence 값 체크
        if conf < 0.65:
            continue
        x1, y1, x2, y2 = box
        confidence = conf
        detected_class = cls
        name = names[int(cls)]

        if name == 'container':
            container_count += 1
            box_color = (0, 255, 0)  # 컨테이너 색상 (녹색)
            center_color = (0, 255, 255)  # 중심 좌표 색상 (노란색)
            center_radius = 10  # 컨테이너 중심 좌표 원의 크기
            text_color = (0, 255, 0)  # 컨테이너 텍스트 색상 (녹색)
            text_scale = 1.0  # 컨테이너 텍스트 크기
            text_thickness = 2  # 컨테이너 텍스트 두께
            box_thickness = 3  # 컨테이너 박스 두께
            background_color = (0, 255, 255)  # 컨테이너 텍스트 배경 색상 (노란색)

        elif name == 'corner-casting':
            corner_casting_count += 1
            box_color = (0, 0, 255)  # 코너 캐스팅 색상 (빨강)
            center_color = (255, 0, 0)  # 중심 좌표 색상 (파란색)
            center_radius = 5  # 코너 캐스팅 중심 좌표 원의 크기
            text_color = (0, 0, 255)  # 코너 캐스팅 텍스트 색상 (빨강)
            text_scale = 0.8  # 코너 캐스팅 텍스트 크기
            text_thickness = 2  # 코너 캐스팅 텍스트 두께
            box_thickness = 2  # 코너 캐스팅 박스 두께
            background_color = (255, 255, 255)  # 코너 캐스팅 텍스트 배경 색상 (흰색)

        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        detection_results['detections'].append({
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'x_center': x_center,
            'y_center': y_center,
            'class': name,
            'confidence': confidence
        })

        # 바운딩 박스와 중앙 좌표를 그립니다.
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, box_thickness)

        cv2.circle(frame, (int(x_center), int(y_center)), center_radius, center_color, -1)
        cv2.putText(frame, f"({int(x_center)}, {int(y_center)})", (int(x_center), int(y_center) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, center_color, text_thickness)

        # 바운딩 박스 위에 텍스트 표시 (ID, 클래스 이름, 정확도)
        label = f'id:{i} {name} {confidence:.2f}'
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)[0]
        label_x = int(x1)

        if name == 'container':
            label_y = int(y1) - 10 if y1 > 10 else int(y1) + label_size[1] + 10
        elif name == 'corner-casting':
            label_y = int(y2) + label_size[1] + 10 if y2 < frame.shape[0] - 10 else int(y2) - 10

        cv2.rectangle(frame, (label_x, label_y - label_size[1] - 2), (label_x + label_size[0], label_y + 2),
                      background_color, cv2.FILLED)
        cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 0), text_thickness)

    results_data.append(detection_results)
    print(
        f"{frame_count}: {frame.shape[1]}x{frame.shape[0]} {container_count} containers, {corner_casting_count} corner-castings")

    containers = []
    corner_castings = []

    for detection in detection_results['detections']:
        class_name = detection["class"]
        x_center = detection["x_center"]
        y_center = detection["y_center"]

        if class_name == "container":
            containers.append(detection)
        elif class_name == "corner-casting":
            corner_castings.append((x_center, y_center))

    container_castings = []

    for container in containers:
        x1, y1 = container["x1"], container["y1"]
        x2, y2 = container["x2"], container["y2"]

        corners = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
        detected_corners = []

        if corner_castings:
            for corner in corners:
                closest_corner_casting = min(corner_castings, key=lambda cc: distance(corner, cc))
                detected_corners.append(closest_corner_casting)

            # 컨테이너 상단 코너와 상단에 위치한 코너 캐스팅을 비교하여 정렬 여부 확인
            upper_corners = [(x1, y1), (x2, y1)]
            for upper_corner in upper_corners:
                upper_closest_corner_casting = min([cc for cc in corner_castings if cc[1] < upper_corner[1]],
                                                   key=lambda cc: distance(upper_corner, cc), default=None)
                if upper_closest_corner_casting:
                    detected_corners.append(upper_closest_corner_casting)

        else:
            detected_corners = [None] * 4

        container_castings.append(detected_corners)

    # 추가: 컨테이너의 중심 좌표를 이용한 정렬 탐지
    container_centers = [(container["x_center"], container["y_center"]) for container in containers]
    if len(container_centers) > 1:  # 두 개 이상의 컨테이너가 있는지 확인
        for i in range(len(container_centers) - 1):
            if abs(container_centers[i][0] - container_centers[i + 1][0]) > 10:
                is_aligned = False
                break
        else:
            is_aligned = True
    else:
        is_aligned = True

    if is_aligned:
        alignment_text = "All Containers Aligned"
        alignment_color = (0, 255, 0)
    else:
        alignment_text = "Containers Not Aligned"
        alignment_color = (0, 0, 255)

    detection_results['aligned'] = is_aligned

    # 주석이 달린 프레임에 정렬 여부 텍스트 추가
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 3
    text_size = cv2.getTextSize(alignment_text, font, font_scale, font_thickness)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = text_size[1] + 20

    cv2.putText(frame, alignment_text, (text_x, text_y), font, font_scale, alignment_color, font_thickness, cv2.LINE_AA)

    # 비디오에 프레임을 저장합니다.
    out.write(frame)

    # 주석이 달린 프레임을 화면에 표시합니다.
    cv2.imshow("YOLOv8", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
out.release()
cv2.destroyAllWindows()

with open('new.json', 'w') as json_file:
    json.dump(results_data, json_file, indent=4)