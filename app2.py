import cv2
import numpy as np
import time
import tflite_runtime.interpreter as tflite

# =============================
# Configuration
# =============================
print("--- Configuration System ---")
MODEL_PATH = "model.tflite"
CONF_THRESHOLD = 0.5
COOLDOWN = 5

# =============================
# Chargement du modèle TFLite
# =============================
print("Chargement TFLite ...")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =============================
# Initialisation caméra
# =============================
print("Initialisation de la camera (/dev/video0) ...")
cap = cv2.VideoCapture(0) # standard for /dev/video0

if not cap.isOpened():
    print("CRITICAL ERROR: Could not open /dev/video0. Check permissions or connection.")
    exit(1)

# =============================
# DEBUG: Startup Screenshot
# =============================
print("DEBUG: Capturing initial test image to /tmp/startup_test.jpg...")
ret, test_frame = cap.read()
if ret:
    # If camera is mono, converting to BGR so we can save a standard JPG
    if len(test_frame.shape) == 2:
        test_frame = cv2.cvtColor(test_frame, cv2.COLOR_GRAY2BGR)
    cv2.imwrite("/tmp/startup_test.jpg", test_frame)
    print("DEBUG: Startup image saved successfully.")
else:
    print("DEBUG ERROR: Failed to capture startup image.")

last_sent = 0

# =============================
# Boucle principale
# =============================
print("System Ready. Starting detection loop...")

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.1)
            continue

        # 1. Adapt for Monochrome Camera (OV9281)
        # Ensure image has 3 channels for the YOLO model
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # 2. Pre-processing
        input_h = input_details[0]['shape'][1]
        input_w = input_details[0]['shape'][2]

        img = cv2.resize(frame, (input_w, input_h))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        # 3. Inference
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

        # 4. Handle YOLOv8/v10 Output Formatting
        # Most TFLite exports return [1, 84, 8400]
        detections = interpreter.get_tensor(output_details[0]['index'])
        output = np.squeeze(detections) # Remove batch dim -> [84, 8400]
        output = output.T               # Transpose -> [8400, 84]

        person_detected = False

        # 5. Detection Check
        for det in output:
            # det[0:4] = box, det[4:] = scores
            scores = det[4:]
            conf = np.max(scores)
            cls_id = np.argmax(scores)

            if conf > CONF_THRESHOLD and int(cls_id) == 0: # 0 = 'person'
                person_detected = True
                print(f"--- [DEBUG] MATCH FOUND: Person (Conf: {conf:.2f}) ---")
                break

        # 6. Action on detection
        if person_detected and (time.time() - last_sent > COOLDOWN):
            timestamp_str = time.strftime("%H:%M:%S")
            timestamp_file = time.strftime("%H-%M-%S")

            message = f"HUMAN DETECTED | Time: {timestamp_str}"
            print(f"CAPTURING: {message}")

            # Watermarking
            overlay = frame.copy()
            cv2.putText(overlay, message, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2, cv2.LINE_AA)

            frame_final = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            filename = f"/tmp/detection_{timestamp_file}.jpg"
            cv2.imwrite(filename, frame_final)

            last_sent = time.time()

except KeyboardInterrupt:
    print("Arrêt propre du programme")
    cap.release()
