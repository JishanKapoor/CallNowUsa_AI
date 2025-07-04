# from flask import Flask, request, jsonify, render_template
# import sys
# import os
# import threading
# import logging
# from ai import SMSAssistant

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Initialize Flask app
# app = Flask(__name__)

# # Initialize SMSAssistant
# assistant = SMSAssistant()

# # Store output messages to capture print statements
# class OutputCapture:
#     def __init__(self):
#         self.messages = []

#     def write(self, message):
#         if message.strip():
#             self.messages.append(message.strip())

#     def flush(self):
#         pass

#     def get_messages(self):
#         return [msg for msg in self.messages if not (
#             msg.startswith('Parsed Command:') or
#             msg.startswith('🧪') or
#             'Adding contact: alias=' in msg
#         )]

#     def clear(self):
#         self.messages = []

# output_capture = OutputCapture()

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/api/command', methods=['POST'])
# def handle_command():
#     try:
#         data = request.get_json()
#         command = data.get('command', '').strip()
#         if not command:
#             return jsonify({'status': 'error', 'message': 'Empty command'}), 400

#         # Redirect print output to capture
#         sys.stdout = output_capture
#         output_capture.clear()

#         # Process the command
#         for cmd in assistant.parser.parse(command):
#             assistant._dispatch(cmd)

#         # Restore stdout and get filtered messages
#         sys.stdout = sys.__stdout__
#         messages = output_capture.get_messages()

#         return jsonify({
#             'status': 'success',
#             'messages': messages
#         })

#     except Exception as e:
#         sys.stdout = sys.__stdout__
#         logger.error(f"Error processing command: {str(e)}")
#         return jsonify({'status': 'error', 'message': str(e)}), 500

# def run_assistant():
#     # Start the monitor in a separate thread
#     assistant.monitor.start()

# if __name__ == '__main__':
#     # Start the monitor thread
#     monitor_thread = threading.Thread(target=run_assistant, daemon=True)
#     monitor_thread.start()

#     # Run Flask app
#     app.run(debug=True, host='0.0.0.0', port=5000)
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
import sys
import os
import threading
import logging
from ai import SMSAssistant

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Replace with a secure key in production

# Initialize SMSAssistant
assistant = SMSAssistant()

# Store output messages to capture print statements
class OutputCapture:
    def __init__(self):
        self.messages = []

    def write(self, message):
        if message.strip():
            self.messages.append(message.strip())

    def flush(self):
        pass

    def get_messages(self):
        return [msg for msg in self.messages if not (
                msg.startswith('Parsed Command:') or
                msg.startswith('🧪') or
                'Adding contact: alias=' in msg
        )]

    def clear(self):
        self.messages = []


output_capture = OutputCapture()

@app.route('/')
def index():
    # Render the index page regardless of the login status
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        # Check credentials
        if email == 'jishankapoor602@gmail.com' and password == 'jishan1010':
            session['logged_in'] = True
            return redirect(url_for('ai'))  # Redirect to ai.html when logged in
        else:
            flash('Invalid email or password. Please try again.', 'error')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/ai')
def ai():
    if 'logged_in' in session and session['logged_in']:
        return render_template('ai.html')  # Render ai.html only if logged in
    return redirect(url_for('login'))  # If not logged in, redirect to login

@app.route('/api/command', methods=['POST'])
def handle_command():
    try:
        data = request.get_json()
        command = data.get('command', '').strip()
        if not command:
            return jsonify({'status': 'error', 'message': 'Empty command'}), 400

        # Redirect print output to capture
        sys.stdout = output_capture
        output_capture.clear()

        # Process the command
        for cmd in assistant.parser.parse(command):
            assistant._dispatch(cmd)

        # Restore stdout and get filtered messages
        sys.stdout = sys.__stdout__
        messages = output_capture.get_messages()

        return jsonify({
            'status': 'success',
            'messages': messages
        })

    except Exception as e:
        sys.stdout = sys.__stdout__
        logger.error(f"Error processing command: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


def run_assistant():
    # Start the monitor in a separate thread
    assistant.monitor.start()


if __name__ == '__main__':
    # Start the monitor thread
    monitor_thread = threading.Thread(target=run_assistant, daemon=True)
    monitor_thread.start()

    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=8000)
