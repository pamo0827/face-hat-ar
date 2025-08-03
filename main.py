import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

hat = cv2.imread("minecraft_hat_front.png", cv2.IMREAD_UNCHANGED)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

prev_faces = []
face_tracking_frames = 0
TRACKING_THRESHOLD = 5

def is_similar_face(face1, face2, threshold=50):
    """2つの顔領域が似ているかチェック"""
    x1, y1, w1, h1 = face1
    x2, y2, w2, h2 = face2
    
    center1 = (x1 + w1//2, y1 + h1//2)
    center2 = (x2 + w2//2, y1 + h1//2)
    distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    return distance < threshold

def stabilize_faces(current_faces, prev_faces):
    """顔検出結果を安定化"""
    if len(prev_faces) == 0:
        return current_faces
    
    stable_faces = []
    for curr_face in current_faces:
        for prev_face in prev_faces:
            if is_similar_face(curr_face, prev_face):
                x1, y1, w1, h1 = curr_face
                x2, y2, w2, h2 = prev_face
                stable_face = (
                    int((x1 + x2) / 2),
                    int((y1 + y2) / 2),
                    int((w1 + w2) / 2),
                    int((h1 + h2) / 2)
                )
                stable_faces.append(stable_face)
                break
        else:
            stable_faces.append(curr_face)
    
    return stable_faces

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.equalizeHist(gray)
    
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.05,
        minNeighbors=6,
        minSize=(50, 50),
        maxSize=(300, 300)
    )
    
    if len(faces) > 0:
        faces = stabilize_faces(faces, prev_faces)
        prev_faces = faces
        face_tracking_frames += 1
    else:
        face_tracking_frames = max(0, face_tracking_frames - 2)
    
    if face_tracking_frames >= TRACKING_THRESHOLD:
        for (x, y, w, h) in faces:
            hat_width = int(w * 1.8)
            hat_height = int(h * 0.9)
            hat_resized = cv2.resize(hat, (hat_width, hat_height))
            
            hat_x_offset = x - int((hat_width - w) / 2)
            hat_y_offset = y - int(h * 0.4)
            
            for i in range(hat_resized.shape[0]):
                for j in range(hat_resized.shape[1]):
                    y_pos = hat_y_offset + i
                    x_pos = hat_x_offset + j

                    if (0 <= y_pos < frame.shape[0] and 
                        0 <= x_pos < frame.shape[1] and 
                        hat_resized.shape[2] == 4):
                        
                        alpha = hat_resized[i, j, 3] / 255.0
                        if alpha > 0.1:
                            for c in range(3):
                                frame[y_pos, x_pos, c] = (
                                    alpha * hat_resized[i, j, c] +
                                    (1 - alpha) * frame[y_pos, x_pos, c]
                                )
    
    cv2.putText(frame, f"Faces: {len(faces)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Stable: {face_tracking_frames >= TRACKING_THRESHOLD}", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hat AR - Improved Detection", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()