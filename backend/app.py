import face_recognition
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import subprocess
import json
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

# User model
class User(db.Model):
    username = db.Column(db.String(80), primary_key=True)
    face_encoding = db.Column(db.PickleType, nullable=False)
    daily_streak = db.Column(db.Integer, default=0)
    last_active = db.Column(db.Date, default=datetime.utcnow)
    overall_accuracy = db.Column(db.Float, default=0.0)
    exercises_done = db.Column(db.Integer, default=0)
    recently_completed = db.Column(db.PickleType, default=[])
    recommendations = db.Column(db.PickleType, default=[])
    accuracy_increments = db.Column(db.Integer, default=0)
    total_exercises = db.Column(db.Integer, default=0)

    def to_dict(self):
        return {
            'username': self.username,
            'stats': [
                {
                    'title': 'Daily Streak',
                    'description': 'Flexy every day!',
                    'number': self.daily_streak,
                },
                {
                    'title': 'Overall Accuracy',
                    'description': 'Not flexy, you know it.',
                    'number': self.overall_accuracy,
                },
                {
                    'title': 'Exercises Done',
                    'description': 'Super flexy!',
                    'number': self.exercises_done,
                },
            ],
            'recentlyCompleted': self.recently_completed,
            'recommendations': self.recommendations,
            'accuracyIncrements': self.accuracy_increments,
            'totalExercises': self.total_exercises,
        }

# Create the database tables
with app.app_context():
    db.create_all()

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
