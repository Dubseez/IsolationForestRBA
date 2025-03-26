import pandas as pd
import random
import datetime
from timezonefinder import TimezoneFinder

def generate_random_lat_long():
    """Generates a random latitude and longitude far from previous entries."""
    return round(random.uniform(-90, 90), 6), round(random.uniform(-180, 180), 6)

def generate_random_ip():
    """Generates a random IP address."""
    return f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 255)}"

def generate_new_device():
    """Randomly selects a new device type."""
    devices = ["Windows 10", "MacBook Pro", "iPhone 14", "Samsung Galaxy S23", "Linux Ubuntu"]
    return random.choice(devices)

def add_6th_entry(file_path, output_path):
    tf = TimezoneFinder()
    
    # Load dataset
    df = pd.read_csv(file_path)
    
    new_entries = []
    
    for user in df['user_id'].unique():
        user_data = df[df['user_id'] == user].sort_values(by='login_time')
        last_entry = user_data.iloc[-1]
        
        # Generate new lat-long drastically different
        new_lat, new_long = generate_random_lat_long()
        
        # Determine new timezone
        new_timezone = tf.timezone_at(lng=new_long, lat=new_lat) or "Unknown"
        
        # Create new entry
        new_entry = {
            'user_id': user,
            'ip_address': generate_random_ip(),
            'latitude': new_lat,
            'longitude': new_long,
            'timezone': new_timezone,
            'device_info': generate_new_device(),
            'typing_speed': round(random.uniform(20, 30), 2),  # Above 20
            'mouse_speed': round(random.uniform(3000, 5000), 2),  # Above 3000
            'login_time': (datetime.datetime.strptime(last_entry['login_time'], "%d-%m-%Y %H:%M") + datetime.timedelta(hours=1)).strftime("%d-%m-%Y %H:%M")
        }
        
        new_entries.append(new_entry)
    
    # Convert new entries to DataFrame and append to original
    new_df = pd.DataFrame(new_entries)
    updated_df = pd.concat([df, new_df], ignore_index=True)
    
    # Sort by user_id and login_time
    updated_df = updated_df.sort_values(by=['user_id', 'login_time']).reset_index(drop=True)
    
    # Save updated data
    updated_df.to_csv(output_path, index=False)
    print(f"Updated data saved to {output_path}")

# Run script
add_6th_entry("augmented_login_data_v4.csv", "IFdata.csv")