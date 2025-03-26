from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import joblib
import numpy as np
from geopy.distance import geodesic
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:root@localhost/risk_auth_iforest_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Load trained Isolation Forest model and preprocessing objects
iso_forest = joblib.load("isolation_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
ip_frequencies = joblib.load("ip_frequencies.pkl")

# Define the database model
class LoginAttempt(db.Model):
    __tablename__ = "login_attempts"  # Ensure correct table name
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.String(255), nullable=False)
    ip_address = db.Column(db.String(50), nullable=False)
    latitude = db.Column(db.Float, nullable=False, default=0.0)
    longitude = db.Column(db.Float, nullable=False, default=0.0)
    timezone = db.Column(db.String(50))
    device_info = db.Column(db.String(255), default='Unknown')
    typing_speed = db.Column(db.Float, default=0.0)
    mouse_speed = db.Column(db.Float, default=0.0)
    geo_velocity = db.Column(db.Float, default=0.0)
    login_time = db.Column(db.DateTime, default=datetime.utcnow)

# Function to calculate geo-velocity
def calculate_geo_velocity(prev_lat, prev_lon, prev_time, curr_lat, curr_lon, curr_time):
    try:
        if prev_lat is None or prev_lon is None or prev_time is None:
            return 0.0
        prev_location = (prev_lat, prev_lon)
        current_location = (curr_lat, curr_lon)
        time_diff = (curr_time - prev_time).total_seconds() / 3600.0  # Convert seconds to hours
        if time_diff <= 0:
            return 0.0
        distance = geodesic(prev_location, current_location).km
        return distance / time_diff
    except Exception as e:
        print(f"Error calculating geo-velocity: {e}")
        return 0.0

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        user_id = data.get("user_id")
        ip_address = data.get("ip_address")
        latitude = float(data.get("latitude", 0.0))
        longitude = float(data.get("longitude", 0.0))
        timezone = data.get("timezone", "Unknown")
        device_info = data.get("device_info", "Unknown")
        typing_speed = float(data.get("typing_speed", 0.0))
        mouse_speed = float(data.get("mouse_speed", 0.0))
        login_time = datetime.utcnow()
        login_hour = login_time.hour  # Extract hour from login time

        # Get previous login attempt for geo-velocity calculation
        prev_attempt = LoginAttempt.query.filter_by(user_id=user_id).order_by(LoginAttempt.login_time.desc()).first()
        prev_lat = prev_attempt.latitude if prev_attempt else None
        prev_lon = prev_attempt.longitude if prev_attempt else None
        prev_time = prev_attempt.login_time if prev_attempt else None

        geo_velocity = calculate_geo_velocity(prev_lat, prev_lon, prev_time, latitude, longitude, login_time)

        # Encode categorical features safely
        ip_freq = ip_frequencies.get(ip_address, 0.0001)  # Default frequency for new IPs

        if "timezone" in label_encoders:
            timezone = label_encoders["timezone"].transform([timezone])[0] if timezone in label_encoders["timezone"].classes_ else label_encoders["timezone"].transform(["Unknown"])[0]

        if "device_info" in label_encoders:
            device_info = label_encoders["device_info"].transform([device_info])[0] if device_info in label_encoders["device_info"].classes_ else label_encoders["device_info"].transform(["Unknown"])[0]

        # Prepare feature vector
        feature_vector = np.array([[latitude, longitude, typing_speed, mouse_speed, geo_velocity, login_hour, ip_freq]])
        feature_vector = scaler.transform(feature_vector)  # Ensure proper scaling

        # Get risk score
        risk_score = -iso_forest.decision_function(feature_vector)[0]

        # Determine risk decision
        if geo_velocity > 1000:
            risk_decision = "Block (Unrealistic Travel Speed)"
        elif risk_score < -0.05:
            risk_decision = "Allow"
        elif -0.05 <= risk_score <= 0:
            risk_decision = "MFA"
        else:
            risk_decision = "Block"

        # Only save to database if decision is "Allow"
        if risk_decision == "Allow":
            new_attempt = LoginAttempt(
                user_id=user_id,
                ip_address=ip_address,
                latitude=latitude,
                longitude=longitude,
                timezone=data.get("timezone", "Unknown"),
                device_info=data.get("device_info", "Unknown"),
                typing_speed=typing_speed,
                mouse_speed=mouse_speed,
                geo_velocity=geo_velocity,
                login_time=login_time
            )
            db.session.add(new_attempt)
            db.session.commit()

        return jsonify({
            "risk_score": risk_score,
            "geo_velocity": geo_velocity,
            "risk_decision": risk_decision
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)