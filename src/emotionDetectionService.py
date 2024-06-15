import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import traceback

# Load the saved model
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, 'D:\Mihin\Models\Emotion Recognition Model\Emotion_detection_with_CNN\Emotion_detection_with_CNN-main\model\emotion_model(AB).h5')
emotion_model = load_model(model_path)
print("Loaded model from disk")

# Emotion labels dictionary
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad"}

def predictEmotion(imagefile):
    try:
        # Read the image file and convert to OpenCV format
        file_bytes = np.frombuffer(imagefile.read(), np.uint8)
        captured_frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if captured_frame is None:
            print("Error: captured_frame is None. Image might not have been read correctly.")
            return "Image read error"
        
        # Resize and convert the captured image to grayscale
        gray_frame = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2GRAY)
        
        # Save the image for debugging purposes
        cv2.imwrite('debug_image.jpg', gray_frame)
        
        # Load the Haar cascade for face detection
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        if face_detector.empty():
            print("Error: Haar cascade not loaded correctly.")
            return "Haar cascade load error"
        
        # Detect faces in the image
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
        
        if len(num_faces) == 0:
            print("No faces detected.")
            return "No faces detected"
        
        # Process each detected face
        for (x, y, w, h) in num_faces:
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # Predict the emotion
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            predictedEmotion = emotion_dict[maxindex]
            print(f"Detected face. Predicted emotion: {predictedEmotion}")
        
        return predictedEmotion
    
    except Exception as e:
        traceback.print_exc()
        print("Error! - Predicting Emotion:", e)
        return None
