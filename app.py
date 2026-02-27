import cv2
import numpy as np
import time
import requests
import tflite_runtime.interpreter as tflite

# =============================
# Configuration
# =============================
MODEL_PATH = "model.tflite"            # Chemin vers le modèle TFLite embarqué
# SEND_URL = "http://SERVER_IP:5000/upload"  # URL serveur (désactivée pour le moment)
CONF_THRESHOLD = 0.5                    # Seuil de confiance pour valider détection
COOLDOWN = 5                            # Délai minimum entre deux détections (secondes)

# =============================
# Chargement du modèle TFLite
# =============================
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Récupération des informations d'entrée / sortie du modèle
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =============================
# Initialisation caméra (Raspberry)
# =============================
# /dev/video0 correspond généralement à une caméra USB
# ou à une caméra CSI exposée via V4L2
cap = cv2.VideoCapture("/dev/video0")

last_sent = 0  # Timestamp du dernier envoi

# =============================
# Boucle principale
# =============================
try:
    while True:
        ret, frame = cap.read()  # Capture image caméra
        if not ret:
            time.sleep(0.1)
            continue  # Si échec lecture, on attend légèrement

        # Préparation image pour inférence
        # Resize vers la taille attendue par le modèle
        input_height = input_details[0]['shape'][1]
        input_width = input_details[0]['shape'][2]
        img = cv2.resize(frame, (input_width, input_height))
        img = img.astype(np.float32) / 255.0  # Normalisation [0,1]
        img = np.expand_dims(img, axis=0)     # Ajout dimension batch

        # Exécution inférence
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

        # Récupération des détections
        detections = interpreter.get_tensor(output_details[0]['index'])

        person_detected = False

        # Parcours des objets détectés
        for det in detections[0]:
            x1, y1, x2, y2, conf, cls = det
            # Classe 0 correspond à "person" dans YOLO COCO
            if int(cls) == 0 and conf > CONF_THRESHOLD:
                person_detected = True
                break  # Une seule personne suffit pour déclencher

        # Si personne détectée ET cooldown respecté
        if person_detected and (time.time() - last_sent > COOLDOWN):
            timestamp_str = time.strftime("%H:%M:%S")
            timestamp_file = time.strftime("%H-%M-%S")

            message = f"HUMAN DETECTED | Time: {timestamp_str}"

            # Ajout filigrane texte sur l'image
            overlay = frame.copy()
            cv2.putText(
                overlay,
                message,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

            alpha = 0.6
            frame_watermarked = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            filename = f"/tmp/detection_{timestamp_file}.jpg"

            cv2.imwrite(
                filename,
                frame_watermarked,
                [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            )

            print(message)

            # === Envoi HTTP désactivé (à réactiver si besoin) ===
            # try:
            #     with open(filename, "rb") as f:
            #         requests.post(SEND_URL, files={"file": f}, timeout=2)
            # except Exception as e:
            #     print("Erreur envoi:", e)

            last_sent = time.time()

except KeyboardInterrupt:
    print("Arrêt propre du programme")
    cap.release()
    cv2.destroyAllWindows()