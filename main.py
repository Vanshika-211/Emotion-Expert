from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
import text_analysis
import audio_analysis
import video_analysis

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'your_secret_key'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/text_analysis', methods=['GET', 'POST'])
def text_analysis_route():
    if request.method == 'POST':
        text_input = request.form['input_text']
        pdf_file = request.files.get('pdf_file')

        if text_input:
            emotion_results = text_analysis.perform_analysis(input_text=text_input)
        elif pdf_file:
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(pdf_file.filename))
            pdf_file.save(pdf_path)
            emotion_results = text_analysis.perform_analysis(pdf_path=pdf_path)
        else:
            flash('Please provide either text input or upload a PDF file.')
            return redirect(request.url)
        
        return render_template('results.html', emotion_results=emotion_results, analysis_type='Text Analysis')

    return render_template('text_analysis.html')

@app.route('/audio_analysis', methods=['GET', 'POST'])
def audio_analysis_route():
    if request.method == 'POST':
        audio_file = request.files.get('audio_file')
        if audio_file:
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(audio_file.filename))
            audio_file.save(audio_path)
            emotion_results = audio_analysis.recognize_emotion(audio_path)
            return render_template('results.html', emotion_results=emotion_results, analysis_type='Audio Analysis')
        else:
            flash('Please upload an audio file.')
            return redirect(request.url)

    return render_template('audio_analysis.html')

@app.route('/video_analysis', methods=['GET', 'POST'])
def video_analysis_route():
    if request.method == 'POST':
        video_file = request.files.get('video_file')
        if video_file:
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(video_file.filename))
            video_file.save(video_path)
            emotion_results = video_analysis.analyze_video_emotions(video_path)
            return render_template('results.html', emotion_results=emotion_results, analysis_type='Video Analysis')
        else:
            flash('Please upload a video file.')
            return redirect(request.url)

    return render_template('video_analysis.html')

if __name__ == '__main__':
    app.run(debug=True)