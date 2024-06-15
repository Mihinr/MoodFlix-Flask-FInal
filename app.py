from flask import Flask, jsonify, request

# Import the predictEmotion function from the emotionDetectionService module
from src.emotionDetectionService import predictEmotion

# Initialize app
app = Flask(__name__)

@app.route('/predict/emotion', methods=['POST'])
def predictEmotionController():
    try:
        if 'image' not in request.files:
            return ("No image part in the request", 400)
        
        imagefile = request.files['image']
        
        if imagefile.filename == '': 
            return ("No selected image", 400)
        
        # Call the predictEmotion function from the emotionDetectionService module
        emotion = predictEmotion(imagefile)
        
        if emotion is not None:
            data = {
                "success": True,
                "data": emotion
            }
            return jsonify(data), 200
        else:
            data = {
                "success": False,
                "data": None
            }
            return jsonify(data), 200
        
    except Exception as e:
        print("Error! - Predicting Emotion:", e)
        data = {
            "success": False,
            "data": None
        }
        return jsonify(data), 200


@app.route('/')
def checkFlaskApi():
    return 'Flask APi Working!'


# App main function
if __name__ == '__main__':
    app.run(port=5000, debug=True)
