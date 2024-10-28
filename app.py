from flask import Flask, render_template, request, jsonify
import csv

app = Flask(__name__)

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# API to save the matrix data to a CSV file
@app.route('/save', methods=['POST'])
def save_data():
    data = request.json
    if data is None:
        return jsonify({"status": "error", "message": "No data provided"}), 400

    label = data.get('label')
    matrix = data.get('matrix')

    # Flatten the matrix and prepare it for CSV saving
    flat_matrix = [label] + [cell for row in matrix for cell in row]

    with open('alphabet_data.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(flat_matrix)

    return jsonify({"status": "success", "message": f"Saved label '{label}'"})

if __name__ == '__main__':
    app.run(debug=True)
