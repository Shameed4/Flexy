import subprocess
from flask import Flask

app = Flask(__name__)

@app.route('/run-script', methods=['GET'])
def run_script():
    try:
        # Execute the command to run main.py
        result = subprocess.run(['python3', 'changing_circles.py'], check=True)
        return "done", 200
    except subprocess.CalledProcessError as e:
        # If there's an error running the script, return the error message
        return f"Script failed with error: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)