from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import pickle

app = Flask(__name__)

# load model & fitur
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("features.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Halaman utama
@app.route("/")
def home():
    return """
    <h2>Decision Tree Drug Prediction API 🚀</h2>
    <a href="/form">👉 Coba Prediksi lewat Form</a>
    """

# Halaman form input
@app.route("/form", methods=["GET", "POST"])
def form():
    html = """
    <h2>Prediksi Obat Pasien</h2>
    <form method="post">
      Age: <input type="number" name="Age"><br><br>
      Na_to_K: <input type="number" step="0.01" name="Na_to_K"><br><br>

      Sex:
      <select name="Sex_M">
        <option value="0">Female</option>
        <option value="1">Male</option>
      </select><br><br>

      BP:
      <select name="BP_LOW">
        <option value="0">Not LOW</option>
        <option value="1">LOW</option>
      </select>
      <select name="BP_NORMAL">
        <option value="0">Not NORMAL</option>
        <option value="1">NORMAL</option>
      </select><br><br>

      Cholesterol:
      <select name="Cholesterol_NORMAL">
        <option value="0">HIGH</option>
        <option value="1">NORMAL</option>
      </select><br><br>

      <input type="submit" value="Prediksi">
    </form>
    """

    if request.method == "POST":
        # ambil data dari form
        data = {
                col: [float(request.form.get(col, 0))] if col in ["Age", "Na_to_K"]
                else [int(request.form.get(col, 0))]
                for col in feature_names
            }
        df = pd.DataFrame(data)

        pred = model.predict(df)[0]

        return html + f"<h3>Hasil Prediksi: {pred}</h3>"

    return html

if __name__ == "__main__":
    app.run(debug=True)
