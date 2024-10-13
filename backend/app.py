# app.py

import face_recognition
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import json

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})


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
    print("Sean is a nerd")
    try:
        file = request.files['file']
        username = request.form['username']
        print(f"Received username: {username}")
        print(f"Received file: {file.filename}")
        
        # Load the uploaded image
        img = face_recognition.load_image_file(file)
        face_encodings = face_recognition.face_encodings(img)
        
        if len(face_encodings) == 0:
            print("No face detected.")
            return jsonify({'status': 'fail', 'message': 'No face detected'})

        encoding = face_encodings[0]

        # Store the encoding on the blockchain (or a simulated local storage)
        store_face_on_blockchain(username, encoding)

        return jsonify({'status': 'success', 'message': 'User registered successfully'})
    except Exception as e:
        print(f"Error during registration: {str(e)}")
        return jsonify({'status': 'fail', 'message': 'Registration failed'})


@app.route('/login', methods=['POST'])
def login():
    file = request.files['file']
    username = request.form['username']

    # Load the uploaded image
    img = face_recognition.load_image_file(file)
    face_encodings = face_recognition.face_encodings(img)

    if len(face_encodings) == 0:
        return jsonify({'status': 'fail', 'message': 'No face detected'})

    encoding = face_encodings[0]

    try:
        # Retrieve the stored encoding from the blockchain
        known_encoding = retrieve_face_from_blockchain(username)
    except subprocess.CalledProcessError:
        return jsonify({'status': 'fail', 'message': 'User not found'})

    # Compare the encodings
    matches = face_recognition.compare_faces([known_encoding], encoding)

    if True in matches:
        return jsonify({'status': 'success', 'message': 'Authentication successful'})
    else:
        return jsonify({'status': 'fail', 'message': 'Authentication failed'})

if __name__ == '__main__':
    app.run(debug=True)
