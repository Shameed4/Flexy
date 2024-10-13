import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import subprocess
import json
import face_recognition
from datetime import datetime, timedelta
import numpy as np
import subprocess


app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return "API running"

@app.route('/stretch-circle', methods=['GET'])
def run_circle():
    try:
        # Run the script and capture the output
        result = subprocess.run(
            ['python3', 'stretch/arm/circle.py'],
            check=True,
            capture_output=True,
            text=True
        )

        # Append a message after the script ends
        response_data = {
            'output': result.stdout,
            'message': "Script has completed successfully!"
        }

        return jsonify(response_data), 200

    except subprocess.CalledProcessError as e:
        # Capture and return the error message if the script fails
        error_response = {
            'error': e.stderr,
            'message': f"Script failed with return code: {e.returncode}"
        }
        return jsonify(error_response), 500

@app.route('/stretch-loop', methods=['GET'])
def run_loop():
    try:
        # Run the script and capture the output
        result = subprocess.run(
            ['python3', 'stretch/arm/loop.py'],
            check=True,
            capture_output=True,
            text=True
        )

        # Append a message after the script ends
        response_data = {
            'output': result.stdout,
            'message': "Script has completed successfully!"
        }

        return jsonify(response_data), 200

    except subprocess.CalledProcessError as e:
        # Capture and return the error message if the script fails
        error_response = {
            'error': e.stderr,
            'message': f"Script failed with return code: {e.returncode}"
        }
        return jsonify(error_response), 500

@app.route('/stretch-wave', methods=['GET'])
def run_wave():
    try:
        # Run the script and capture the output
        result = subprocess.run(
            ['python3', 'stretch/arm/wave.py'],
            check=True,
            capture_output=True,
            text=True
        )

        # Append a message after the script ends
        response_data = {
            'output': result.stdout,
            'message': "Script has completed successfully!"
        }

        return jsonify(response_data), 200

    except subprocess.CalledProcessError as e:
        error_response = {
            'error': e.stderr,
            'message': f"Script failed with return code: {e.returncode}"
        }
        return jsonify(error_response), 500

@app.route('/neck-oval', methods=['GET'])
def run_oval():
    try:
        # Run the script and capture the output
        result = subprocess.run(
            ['python3', 'stretch/neck/ovalShape.py'],
            check=True,
            capture_output=True,
            text=True
        )

        # Append a message after the script ends
        response_data = {
            'output': result.stdout,
            'message': "Script has completed successfully!"
        }

        return jsonify(response_data), 200

    except subprocess.CalledProcessError as e:
        # Capture and return the error message if the script fails
        error_response = {
            'error': e.stderr,
            'message': f"Script failed with return code: {e.returncode}"
        }
        return jsonify(error_response), 500

@app.route('/neck-yesno', methods=['GET'])
def run_yesno():
    try:
        # Run the script and capture the output
        result = subprocess.run(
            ['python3', 'stretch/neck/yesOrNo.py'],
            check=True,
            capture_output=True,
            text=True
        )

        # Append a message after the script ends
        response_data = {
            'output': result.stdout,
            'message': "Script has completed successfully!"
        }

        return jsonify(response_data), 200

    except subprocess.CalledProcessError as e:
        # Capture and return the error message if the script fails
        error_response = {
            'error': e.stderr,
            'message': f"Script failed with return code: {e.returncode}"
        }
        return jsonify(error_response), 500

@app.route('/shoulder-grab', methods=['GET'])
def run_grab():
    try:
        # Run the script and capture the output
        result = subprocess.run(
            ['python3', 'stretch/shoulder/grabbingObjects.py'],
            check=True,
            capture_output=True,
            text=True
        )

        # Append a message after the script ends
        response_data = {
            'output': result.stdout,
            'message': "Script has completed successfully!"
        }

        return jsonify(response_data), 200

    except subprocess.CalledProcessError as e:
        # Capture and return the error message if the script fails
        error_response = {
            'error': e.stderr,
            'message': f"Script failed with return code: {e.returncode}"
        }
        return jsonify(error_response), 500

@app.route('/shoulder-reach', methods=['GET'])
def run_reach():
    try:
        # Run the script and capture the output
        result = subprocess.run(
            ['python3', 'stretch/shoulder/objectReaching.py'],
            check=True,
            capture_output=True,
            text=True
        )

        # Append a message after the script ends
        response_data = {
            'output': result.stdout,
            'message': "Script has completed successfully!"
        }

        return jsonify(response_data), 200

    except subprocess.CalledProcessError as e:
        # Capture and return the error message if the script fails
        error_response = {
            'error': e.stderr,
            'message': f"Script failed with return code: {e.returncode}"
        }
        return jsonify(error_response), 500

@app.route('/fitness-circle', methods=['GET'])
def run_fitness_circle():
    try:
        # Run the script and capture the output
        result = subprocess.run(
            ['python3', 'fitness/arm_circles.py'],
            check=True,
            capture_output=True,
            text=True
        )

        # Append a message after the script ends
        response_data = {
            'output': result.stdout,
            'message': "Script has completed successfully!"
        }

        return jsonify(response_data), 200

    except subprocess.CalledProcessError as e:
        # Capture and return the error message if the script fails
        error_response = {
            'error': e.stderr,
            'message': f"Script failed with return code: {e.returncode}"
        }
        return jsonify(error_response), 500

@app.route('/fitness-legs', methods=['GET'])
def run_fitness_legs():
    try:
        # Run the script and capture the output
        result = subprocess.run(
            ['python3', 'fitness/leg_raises.py'],
            check=True,
            capture_output=True,
            text=True
        )

        # Append a message after the script ends
        response_data = {
            'output': result.stdout,
            'message': "Script has completed successfully!"
        }

        return jsonify(response_data), 200

    except subprocess.CalledProcessError as e:
        # Capture and return the error message if the script fails
        error_response = {
            'error': e.stderr,
            'message': f"Script failed with return code: {e.returncode}"
        }
        return jsonify(error_response), 500

@app.route('/fitness-jacks', methods=['GET'])
def run_fitness_jacks():
    try:
        # Run the script and capture the output
        result = subprocess.run(
            ['python3', 'fitness/jumping_jacks.py'],
            check=True,
            capture_output=True,
            text=True
        )

        # Append a message after the script ends
        response_data = {
            'output': result.stdout,
            'message': "Script has completed successfully!"
        }

        return jsonify(response_data), 200

    except subprocess.CalledProcessError as e:
        # Capture and return the error message if the script fails
        error_response = {
            'error': e.stderr,
            'message': f"Script failed with return code: {e.returncode}"
        }
        return jsonify(error_response), 500

def run(model, inputs):
    API_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/b738d494f1baf2b32703c5224c483e9e/ai/run/"
    headers = {"Authorization": "Bearer SwW0_y2MGxqhGj0_wSqEiLDAmqPSGCsPXnbMdKkh"}
    input = {"messages": inputs}
    response = requests.post(f"{API_BASE_URL}{model}", headers=headers, json=input)
    return response.json()

@app.route('/cloudflare_ai', methods=['POST'])
def cloudflare_ai():
    API_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/b738d494f1baf2b32703c5224c483e9e/ai/run/"
    headers = {"Authorization": "Bearer SwW0_y2MGxqhGj0_wSqEiLDAmqPSGCsPXnbMdKkh"}

    def run(model, inputs):
        input = {"messages": inputs}
        response = requests.post(f"{API_BASE_URL}{model}", headers=headers, json=input)
        return response.json()

    inputs = [
    {
        "role": "system",
        "content": "You are a therapist specializing in arthritis treatment. You assist users with exercises and stretches specifically focused on arthritis relief. You will suggest exercises for wrist, shoulders, neck, and workouts like jumping jacks, arm circles, and knee raises. Your responses are concise and focused on arthritis-related benefits."
    },
    {
        "role": "user",
        "content": "Provide suggestions in one-line responses, emphasizing how each exercise helps with arthritis relief. If the user requests something we don't have, respond with 'We don't have an exercise for that yet, but stay tuned for updates!'"
    }
]
    output = run("@cf/meta/llama-3-8b-instruct", inputs)
    print(output)
    return jsonify(output), 200

def store_face_on_blockchain(username, encoding):
    # Convert encoding to JSON string
    encoding_json = json.dumps(encoding.tolist())

    # Call blockchain.js to store the encoding
    subprocess.run(['node', 'blockchain.js', 'store', username, encoding_json], check=True)

def retrieve_face_from_blockchain(username):
    # Call blockchain.js to retrieve the encoding
    result = subprocess.run(['node', 'blockchain.js', 'retrieve', username], capture_output=True, text=True, check=True)

    # Parse the JSON string back to a numpy array
    encoding = np.array(json.loads(result.stdout))
    return encoding

@app.route('/register', methods=['POST'])
def register():
    try:
        file = request.files['file']
        username = request.form['username']

        # Load the uploaded image
        img = face_recognition.load_image_file(file)
        face_encodings = face_recognition.face_encodings(img)

        if len(face_encodings) == 0:
            return jsonify({'status': 'fail', 'message': 'No face detected'})

        encoding = face_encodings[0]

        # Store the encoding on the blockchain
        store_face_on_blockchain(username, encoding)

        # Create a new user with default stats
        new_user = User(
            username=username,
            face_encoding=encoding,
            daily_streak=0,
            last_active=datetime.utcnow(),
            overall_accuracy=0.0,
            exercises_done=0,
            recently_completed=[],
            recommendations=[
                {'title': 'Legs', 'description': 'Strengthen your legs with these exercises.'},
                {'title': 'Shoulder', 'description': 'Improve shoulder mobility and strength.'},
                {'title': 'Ankles', 'description': 'Enhance ankle flexibility and stability.'},
            ],
            accuracy_increments=0,
            total_exercises=0
        )
        db.session.add(new_user)
        db.session.commit()

        return jsonify({'status': 'success', 'message': 'User registered successfully'})
    except Exception as e:
        print(f"Error during registration: {str(e)}")
        return jsonify({'status': 'fail', 'message': 'Registration failed'})

@app.route('/login', methods=['POST'])
def login():
    try:
        file = request.files['file']
        username = request.form['username']

        # Load the uploaded image
        img = face_recognition.load_image_file(file)
        face_encodings = face_recognition.face_encodings(img)

        if len(face_encodings) == 0:
            return jsonify({'status': 'fail', 'message': 'No face detected'})

        encoding = face_encodings[0]

        # Retrieve the stored encoding from the blockchain
        known_encoding = retrieve_face_from_blockchain(username)

        # Compare the encodings
        matches = face_recognition.compare_faces([known_encoding], encoding)

        if True in matches:
            # Update last active date and daily streak
            user = User.query.get(username)
            today = datetime.utcnow().date()
            if user.last_active.date() < today - timedelta(days=1):
                user.daily_streak = 1
            elif user.last_active.date() == today - timedelta(days=1):
                user.daily_streak += 1
            user.last_active = datetime.utcnow()
            db.session.commit()
            return jsonify({'status': 'success', 'message': 'Authentication successful', 'user': user.to_dict()})
        else:
            return jsonify({'status': 'fail', 'message': 'Authentication failed'})
    except subprocess.CalledProcessError:
        return jsonify({'status': 'fail', 'message': 'User not found'})
    except Exception as e:
        print(f"Error during login: {str(e)}")
        return jsonify({'status': 'fail', 'message': 'Login failed'})

@app.route('/user/<username>', methods=['GET'])
def get_user(username):
    user = User.query.get(username)
    if user:
        return jsonify({'status': 'success', 'user': user.to_dict()})
    else:
        return jsonify({'status': 'fail', 'message': 'User not found'})

@app.route('/user/<username>/update', methods=['POST'])
def update_user(username):
    try:
        data = request.json
        user = User.query.get(username)
        if not user:
            return jsonify({'status': 'fail', 'message': 'User not found'})

        # Update user stats based on received data
        user.exercises_done += data.get('exercises_done', 0)
        user.accuracy_increments += data.get('accuracy_increments', 0)
        user.total_exercises += data.get('total_exercises', 0)
        user.overall_accuracy = (user.accuracy_increments / user.total_exercises) if user.total_exercises > 0 else 0.0
        user.recently_completed.extend(data.get('recently_completed', []))

        db.session.commit()
        return jsonify({'status': 'success', 'message': 'User stats updated', 'user': user.to_dict()})
    except Exception as e:
        print(f"Error updating user: {str(e)}")
        return jsonify({'status': 'fail', 'message': 'Update failed'})


if __name__ == '__main__':
    app.run(debug=True)
