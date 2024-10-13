from flask import Flask, jsonify
from flask_cors import CORS
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

if __name__ == '__main__':
    app.run(debug=True)
