import cv2
import numpy as np
import time
import requests
import tensorflow as tf
import os

# === Configuration ===
MODEL_PATH = "model.tflite"           # Chemin du modèle TFLite
SEND_URL = "http://SERVER_IP:5000/upload"  # Serveur réception image
CONF_THRESHOLD = 0.05  # seuil abaissé pour debug
COOLDOWN = 5                           # Temps minimum entre deux messages (sec)
DOWNLOADS_PATH = os.path.expanduser("~/Downloads")

# === Chargement modèle TFLite ===
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Récupération taille entrée modèle (ex: 640x640)
input_shape = input_details[0]['shape']
model_h, model_w = input_shape[1], input_shape[2]
print(f"Model input size: {model_w}x{model_h}")

# === Initialisation webcam Mac ===
cap = cv2.VideoCapture(0)

last_sent = 0
prev_time = time.time()

# === Boucle principale ===
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Préparation image pour modèle
    img = cv2.resize(frame, (model_w, model_h))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Inférence
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    detections = interpreter.get_tensor(output_details[0]['index'])

    # Debug avancé (affiché une seule fois)
    if 'debug_printed' not in globals():
        print("Detections shape:", detections.shape)
        print("Sample values:", detections.flatten()[:20])
        debug_printed = True

    person_detected = False
    best_conf = 0.0
    best_box = None

    # YOLOv8 TFLite output souvent shape (1, 84, N) → on transpose
    output = detections[0]

    if output.shape[0] == 84:
        output = output.T  # (N, 84)

    for det in output:
        x, y, w, h = det[0:4]
        class_scores = det[4:]  # YOLOv8 TFLite export: no separate obj_conf

        cls = np.argmax(class_scores)
        conf = class_scores[cls]  # confidence already includes objectness

        if cls == 0 and conf > best_conf:
            x1 = int((x - w / 2) * frame.shape[1])
            y1 = int((y - h / 2) * frame.shape[0])
            x2 = int((x + w / 2) * frame.shape[1])
            y2 = int((y + h / 2) * frame.shape[0])

            x1 = max(0, min(frame.shape[1], x1))
            y1 = max(0, min(frame.shape[0], y1))
            x2 = max(0, min(frame.shape[1], x2))
            y2 = max(0, min(frame.shape[0], y2))

            best_conf = conf
            best_box = (x1, y1, x2, y2)

    if best_box and best_conf > 0.3:
        x1, y1, x2, y2 = best_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        if best_conf > CONF_THRESHOLD:
            person_detected = True

            # Création message filigrane
            timestamp = time.strftime("%H:%M:%S")
            watermark = f"HUMAN DETECTED | Conf: {best_conf:.2f} | {timestamp}"

            # Overlay semi-transparent
            overlay = frame.copy()
            cv2.putText(overlay, watermark,
                        (50, frame.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2)

            alpha = 0.6
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    if person_detected and (time.time() - last_sent > COOLDOWN):
        timestamp = time.strftime("%H-%M-%S")
        print(f"HUMAN DETECTED | Conf: {best_conf:.2f} | Time: {timestamp}")
        filename = f"detection_{timestamp}.jpg"
        filepath = os.path.join(DOWNLOADS_PATH, filename)
        cv2.imwrite(filepath, frame)
        last_sent = time.time()

    # Calcul FPS
    fps = 1 / (time.time() - prev_time)
    prev_time = time.time()

    # Affichage FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Affichage fenêtre
    cv2.imshow("Detection", frame)

    # Quitter avec ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Libération ressources
cap.release()
cv2.destroyAllWindows()