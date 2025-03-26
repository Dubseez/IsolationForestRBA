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
    __tablename__ = "login_attempts"
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
        time_diff = (curr_time - prev_time).total_seconds() / 3600.0  # Convert to hours
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
        login_hour = login_time.hour

        prev_attempt = LoginAttempt.query.filter_by(user_id=user_id).order_by(LoginAttempt.login_time.desc()).first()
        changed_features = []
        error_score = 0

        if prev_attempt:
            if ip_address != prev_attempt.ip_address:
                error_score += 2
                changed_features.append("IP Address")
            if device_info != prev_attempt.device_info:
                error_score += 3
                changed_features.append("Device")
            if timezone != prev_attempt.timezone:
                error_score += 3
                changed_features.append("Timezone")
            if (latitude, longitude) != (prev_attempt.latitude, prev_attempt.longitude):
                error_score += 5
                changed_features.append("Location")
        
        geo_velocity = calculate_geo_velocity(
            prev_attempt.latitude if prev_attempt else None,
            prev_attempt.longitude if prev_attempt else None,
            prev_attempt.login_time if prev_attempt else None,
            latitude, longitude, login_time
        )

        # Block if geo-velocity exceeds 1000 km/h
        if geo_velocity > 1000:
            return jsonify({
                "isolation_forest_risk_score": "N/A",
                "feature_change_risk_score": error_score,
                "changed_features": changed_features,
                "total_risk_score": "Blocked due to extreme geo-velocity",
                "geo_velocity": geo_velocity,
                "risk_decision": "Block"
            })

        ip_freq = ip_frequencies.get(ip_address, 0.0001)

        if "timezone" in label_encoders and timezone in label_encoders["timezone"].classes_:
            timezone = label_encoders["timezone"].transform([timezone])[0]
        else:
            timezone = label_encoders["timezone"].transform(["Unknown"])[0]

        if "device_info" in label_encoders and device_info in label_encoders["device_info"].classes_:
            device_info = label_encoders["device_info"].transform([device_info])[0]
        else:
            device_info = label_encoders["device_info"].transform(["Unknown"])[0]

        feature_vector = np.array([[latitude, longitude, typing_speed, mouse_speed, geo_velocity, login_hour, ip_freq]])
        feature_vector = scaler.transform(feature_vector)
        risk_score = -iso_forest.decision_function(feature_vector)[0]

        # Always calculate total risk score
        total_risk_score = error_score - risk_score

        if not changed_features:  
            # No changes → Use isolation model thresholds
            if risk_score < -0.11:
                risk_decision = "Allow"
            elif -0.11 <= risk_score <= -0.05:
                risk_decision = "MFA"
            else:
                risk_decision = "Block"
        else:
            # Changes detected → Use total risk score
            if total_risk_score >= 8:
                risk_decision = "Block"
            elif total_risk_score >= 4:
                risk_decision = "MFA"
            else:
                risk_decision = "Allow"

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
            "isolation_forest_risk_score": risk_score,
            "feature_change_risk_score": error_score,
            "changed_features": changed_features,
            "total_risk_score": total_risk_score,  
            "geo_velocity": geo_velocity,
            "risk_decision": risk_decision
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
