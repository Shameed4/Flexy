import subprocess
from flask import Flask

app = Flask(__name__)


@app.route('/run-script', methods=['GET'])
def run_script():
    try:
        # Run the script and capture the output
        result = subprocess.run(
            ['python3', 'changing_circles.py'],
            check=True,
            capture_output=True,  # Capture stdout and stderr
            text=True  # Return output as string instead of bytes
        )

        # Append a message after the script ends
        response = f"{result.stdout}\nScript has completed successfully!"

        return response, 200

    except subprocess.CalledProcessError as e:
        # Capture and return the error message if the script fails
        error_response = f"Script failed with error: {e.stderr}\nReturn code: {e.returncode}"
        return error_response, 500


if __name__ == '__main__':
    app.run(debug=True)
