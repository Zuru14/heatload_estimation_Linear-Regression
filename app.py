from pathlib import Path

from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

# Load model and scaler from the application directory.
model = joblib.load(BASE_DIR / "linear_regression_model.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    input_values = {f"X{index}": "" for index in range(1, 9)}

    if request.method == "POST":
        input_values = {
            f"X{index}": request.form.get(f"X{index}", "").strip()
            for index in range(1, 9)
        }

        try:
            values = [float(input_values[f"X{index}"]) for index in range(1, 9)]
            input_data = np.array(values).reshape(1, -1)
            scaled_data = scaler.transform(input_data)
            prediction = model.predict(scaled_data)[0]
        except (TypeError, ValueError):
            error = "Please enter a valid number in all eight fields."
        except Exception:
            error = "The prediction could not be generated. Please try again."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        input_values=input_values
    )


if __name__ == "__main__":
    app.run(debug=True)