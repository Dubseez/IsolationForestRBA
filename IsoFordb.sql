CREATE DATABASE risk_auth_iforest_db;

USE risk_auth_iforest_db;

CREATE TABLE login_attempts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    ip_address VARCHAR(50) NOT NULL,
    latitude FLOAT NOT NULL DEFAULT 0.0,
    longitude FLOAT NOT NULL DEFAULT 0.0,
    timezone VARCHAR(50),
    device_info VARCHAR(255) DEFAULT 'Unknown',
    typing_speed FLOAT DEFAULT 0.0,
    mouse_speed FLOAT DEFAULT 0.0,
    geo_velocity FLOAT DEFAULT 0.0,
    login_time DATETIME DEFAULT CURRENT_TIMESTAMP
);

select * from login_attempts;