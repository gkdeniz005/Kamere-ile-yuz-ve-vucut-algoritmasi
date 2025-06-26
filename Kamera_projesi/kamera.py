import cv2
import dlib
from imutils import face_utils
import mediapipe as mp
import time

# Yüz ve landmark modeli
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# El ve vücut tanımı için mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Kamerayı başlat
cap = cv2.VideoCapture(0)

face_not_found_time = None
ALERT_DELAY = 3  # saniye olarak yüz bulunmazsa uyarı süresi

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        if face_not_found_time is None:
            face_not_found_time = time.time()
        elif time.time() - face_not_found_time > ALERT_DELAY:
            cv2.putText(frame, "UYARI: Yuz Algilanamiyor!", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        face_not_found_time = None
        for face in faces:
            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            lips_outer = landmarks[48:60]
            lips_inner = landmarks[60:68]

            # Yüz kutusu
            (x, y, w, h) = face_utils.rect_to_bb(face)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (180, 180, 180), 1)

            # Göz çizgileri
            cv2.polylines(frame, [left_eye], isClosed=True, color=(0, 255, 0), thickness=1)
            cv2.polylines(frame, [right_eye], isClosed=True, color=(0, 255, 0), thickness=1)

            # Dudak çizgileri
            cv2.polylines(frame, [lips_outer], isClosed=True, color=(255, 0, 255), thickness=1)
            cv2.polylines(frame, [lips_inner], isClosed=True, color=(255, 0, 255), thickness=1)

    # RGB'ye çevir (mediapipe için)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # El takibi
    result_hands = hands.process(rgb_frame)
    if result_hands.multi_hand_landmarks:
        for hand_landmarks in result_hands.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Vücut takibi
    result_pose = pose.process(rgb_frame)
    if result_pose.pose_landmarks:
        mp_draw.draw_landmarks(frame, result_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Görüntüyü göster
    cv2.imshow("Guvenlik Takibi", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
