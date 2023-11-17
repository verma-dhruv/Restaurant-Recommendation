from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from sklearn.metrics.pairwise import linear_kernel
from recommendation import recommend

app = Flask(__name__)
CORS(app)

localIP = '127.0.0.1'

# Sample data for testing
sample_recommendations = [
    {'name': 'Restaurant 1', 'rate': '4.5', 'location': 'Location 1', 'rest_type': 'Type 1', 'dish_liked': 'Dish 1', 'cuisines': 'Cuisine 1', 'cost': '$$'},
    {'name': 'Restaurant 2', 'rate': '4.0', 'location': 'Location 2', 'rest_type': 'Type 2', 'dish_liked': 'Dish 2', 'cuisines': 'Cuisine 2', 'cost': '$'},
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    user_input = request.json.get('user_input', '')
    recommendations = recommend(user_input)
    return jsonify(recommendations.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
