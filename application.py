from flask import Flask, render_template, request
from pipeline.prediction_pipeline import get_hybrid_recommendation

# Initialize Flask application
app = Flask(__name__)
@app.route('/', methods = ['GET', 'POST'])  ## GET: to render the form, POST: to process the form data



def home():
    recommendations = None

    if request.method == 'POST':
        try:
            user_id = int(request.form['user_id'])
            recommendations = get_hybrid_recommendation(user_id, n= 10)

        except Exception as e:
            print(f"An error occurred: {e}")

        
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0' , port = 5000)  # Set debug=True for development; change to False in production
