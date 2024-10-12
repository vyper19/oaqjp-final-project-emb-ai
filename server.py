"""
Flask-based API to analyze emotions from provided text.
Uses Watson NLP API to predict emotions and return results in JSON format.
"""

from flask import Flask, request, jsonify
import requests  # Third-party import
from EmotionDetection import emotion_detector  # Local application import

app = Flask(__name__)

@app.route('/emotionDetector', methods=['POST'])
def emotion_detector_route():
    """
    Flask route to handle emotion detection requests.
    Receives text input in JSON format and returns emotion scores and dominant emotion.
    """
    try:
        data = request.get_json()
        text_to_analyze = data.get('text', '')

        if not text_to_analyze:
            return jsonify({'error': 'Invalid text! Please try again.'}), 400

        result = emotion_detector(text_to_analyze)

        # If dominant_emotion is None, return an error message
        if result['dominant_emotion'] is None:
            return jsonify({'error': 'Invalid text! Please try again.'}), 400

        # Prepare the output message in the requested format
        response_message = (
            f"For the given statement, the system response is "
            f"'anger': {result['anger']}, 'disgust': {result['disgust']}, "
            f"'fear': {result['fear']}, 'joy': {result['joy']} and "
            f"'sadness': {result['sadness']}. "
            f"The dominant emotion is {result['dominant_emotion']}."
        )

        return jsonify({'response': response_message})

    except requests.exceptions.RequestException as e:
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
