from flask import Flask, jsonify
from Klassifier.predict import make_prediction
#from Klassifier.processing.validation import validate_inputs

app = Flask(__name__)

@app.route('/health',methods=['GET'])
def check_health():
    return 'ok',200

@app.route('/v1/predict/classification', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Step 1: Extract POST data from request body as JSON
        json_data = request.get_json()

        # Step 2: Validate the input using marshmallow schema
        input_data, errors = validate_inputs(input_data=json_data)

        # Step 3: Model prediction
        result = make_prediction(input_data=input_data)

        # Step 4: Convert numpy ndarray to list
        predictions = result.get('predictions').tolist()
        version = result.get('version')

        # Step 5: Return the response as JSON
        return jsonify({'predictions': predictions,
                        'version': version,
                        'errors': errors})

if __name__ == '__main__':
    app.run(debug=True)
