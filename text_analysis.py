import re
from transformers import pipeline
import PyPDF2


# Function to preprocess the text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    return text


# Function to split text into chunks
def split_text_into_chunks(text, chunk_size=100):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield ' '.join(words[i:i + chunk_size])


# Function to analyze emotions in the text
def analyze_emotion(text):
    emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base",
                                return_all_scores=True)

    preprocessed_text = preprocess_text(text)
    text_chunks = list(split_text_into_chunks(preprocessed_text))

    aggregated_scores = {}
    for chunk in text_chunks:
        emotions = emotion_analyzer(chunk)
        for emotion in emotions[0]:
            if emotion['label'] not in aggregated_scores:
                aggregated_scores[emotion['label']] = 0
            aggregated_scores[emotion['label']] += emotion['score']

    num_chunks = len(text_chunks)
    for emotion in aggregated_scores:
        aggregated_scores[emotion] /= num_chunks

    return aggregated_scores


# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page].extract_text()
    return text


# Main function for integrating with Flask
def perform_analysis(input_text=None, pdf_path=None):
    if input_text:
        text = input_text
    elif pdf_path:
        text = extract_text_from_pdf(pdf_path)
    else:
        return None

    # Get the emotion analysis result
    emotion_results = analyze_emotion(text)
    return emotion_results

