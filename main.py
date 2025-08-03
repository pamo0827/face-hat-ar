import cv2
import numpy as np

# Haar Cascade読み込み
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 帽子画像をRGBAで読み込み（透過情報付き）
hat = cv2.imread("minecraft_hat_front.png", cv2.IMREAD_UNCHANGED)

# カメラの初期化を改良
cap = cv2.VideoCapture(0)
# カメラの設定を調整
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# 顔検出の安定化のための変数
prev_faces = []
face_tracking_frames = 0
TRACKING_THRESHOLD = 5  # 5フレーム連続で検出されたら安定とみなす

def is_similar_face(face1, face2, threshold=50):
    """2つの顔領域が似ているかチェック"""
    x1, y1, w1, h1 = face1
    x2, y2, w2, h2 = face2
    
    # 中心点の距離を計算
    center1 = (x1 + w1//2, y1 + h1//2)
    center2 = (x2 + w2//2, y2 + h2//2)
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
                # 前フレームとの平均を取って安定化
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
            # 新しい顔として追加
            stable_faces.append(curr_face)
    
    return stable_faces

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # フレームを反転（鏡像表示）
    frame = cv2.flip(frame, 1)
    
    # グレースケール変換（顔検出の精度向上）
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # ヒストグラム平坦化で照明条件を改善
    gray = cv2.equalizeHist(gray)
    
    # 顔を検出（パラメータを調整して安定性向上）
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.05,  # より細かいスケール調整
        minNeighbors=6,    # より厳しい条件で誤検出を減らす
        minSize=(50, 50),  # 最小サイズを指定
        maxSize=(300, 300) # 最大サイズを指定
    )
    
    # 顔検出結果を安定化
    if len(faces) > 0:
        faces = stabilize_faces(faces, prev_faces)
        prev_faces = faces
        face_tracking_frames += 1
    else:
        face_tracking_frames = max(0, face_tracking_frames - 2)  # 徐々に減少
    
    # 安定して検出されている場合のみ帽子を表示
    if face_tracking_frames >= TRACKING_THRESHOLD:
        for (x, y, w, h) in faces:
            # 帽子をより大きくリサイズ
            hat_width = int(w * 1.8)  # さらに大きく
            hat_height = int(h * 0.9)
            hat_resized = cv2.resize(hat, (hat_width, hat_height))
            
            # 帽子の位置を調整
            hat_x_offset = x - int((hat_width - w) / 2)
            hat_y_offset = y - int(h * 0.4)  # より深く
            
            # アルファブレンディングで帽子を合成
            for i in range(hat_resized.shape[0]):
                for j in range(hat_resized.shape[1]):
                    y_pos = hat_y_offset + i
                    x_pos = hat_x_offset + j

                    # フレームの範囲内か確認
                    if (0 <= y_pos < frame.shape[0] and 
                        0 <= x_pos < frame.shape[1] and 
                        hat_resized.shape[2] == 4):  # RGBA確認
                        
                        alpha = hat_resized[i, j, 3] / 255.0
                        if alpha > 0.1:  # 透明度の閾値を設定
                            for c in range(3):
                                frame[y_pos, x_pos, c] = (
                                    alpha * hat_resized[i, j, c] +
                                    (1 - alpha) * frame[y_pos, x_pos, c]
                                )
    
    # デバッグ情報表示
    cv2.putText(frame, f"Faces: {len(faces)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Stable: {face_tracking_frames >= TRACKING_THRESHOLD}", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 表示
    cv2.imshow("Hat AR - Improved Detection", frame)
    
    # ESCキーまたは'q'キーで終了
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()