from flask import Flask, render_template, request
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load the trained model
model = joblib.load('best_model.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/description')
def description():
    return render_template('description.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
       
        danceability = float(request.form['danceability'])
        energy = float(request.form['energy'])
        loudness = float(request.form['loudness'])
        speechiness = float(request.form['speechiness'])
        acousticness = float(request.form['acousticness'])
        valence = float(request.form['valence'])
        tempo = float(request.form['tempo'])

        
        input_data = np.array([[danceability, energy, loudness, speechiness, acousticness, valence, tempo]])

        
        prediction = model.predict(input_data)[0]

        return render_template('result.html', prediction=prediction)

    return render_template('detect.html')


@app.route('/dashboard')
def dashboard():
    # Sample data
    data = [0.6, 0.7, 0.9, 0.4, 0.8]
    labels = ['Song A', 'Song B', 'Song C', 'Song D', 'Song E']

   
    plt.figure(figsize=(6, 4))
    plt.bar(labels, data, color='skyblue')
    plt.title('Music Popularity Comparison')
    plt.savefig('static/bar_chart.png')
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(labels, data, marker='o', color='lightcoral')
    plt.title('Trend Progression Over Time')
    plt.savefig('static/line_chart.png')
    plt.show()

    return render_template('dashboard.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run()
