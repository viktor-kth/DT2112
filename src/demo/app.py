from flask import Flask, render_template, request, jsonify
import io

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/findmatch', methods=['POST']) 
def findmatch():
    file = request.files['audio']
    buffer = io.BytesIO(file.read())
    buffer.seek(0)
    print(buffer)
    return jsonify({'message': 'File uploaded successfully'})
