from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)
model = pickle.load(open('mlr_model.pkl', 'rb'))

# constants (in usd)
sales_max = 250.0
shipping_cost_max = 16.8
profit_max = 167.5

@app.route("/", methods=['POST', 'GET'])
def home():
    if request.method == 'GET':
        return 'OK'
    else:
        sales = request.json['sales'] / sales_max
        shipping_cost = request.json['shipping_cost'] / shipping_cost_max

        features = [
                    sales, shipping_cost, 
                    request.json['auto_and_accessories'], request.json['electronic'], request.json['fashion'], request.json['home_and_furniture'],
                    request.json['order_priority-critical'], request.json['order_priority-high'], request.json['order_priority-medium'], request.json['order_priority-low']
                   ]
        final_features = [np.array(features)]
        result = model.predict(final_features)

        # in usd
        profit = result[0][0] * profit_max

        return jsonify({ 'profit': profit })

if __name__ == '__main__':
    app.run(debug=True)