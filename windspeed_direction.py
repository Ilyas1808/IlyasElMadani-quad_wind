import requests
import numpy as np
from scipy.interpolate import CubicSpline
from datetime import datetime, timedelta

def fetch_weather_data():
    api_key = "cd28f6d20ad74d998a7aaa0ff83255b5"
    latitude = 50.8371  # Breedtegraad van Etterbeek
    longitude = 4.3887  # Lengtegraad van Etterbeek

    url = f"https://api.weatherbit.io/v2.0/forecast/hourly?lat={latitude}&lon={longitude}&key={api_key}"

    response = requests.get(url)
    data = response.json()

    if "data" in data:
        return data["data"]
    else:
        print("Kon de forecast weergegevens niet vinden in de API-respons.")
        return None

def process_weather_data(forecast_data):
    # data verzamelen
    timestamps = [entry["timestamp_utc"] for entry in forecast_data]
    windsnelheden = [entry["wind_spd"] for entry in forecast_data]
    windrichtingen = [entry["wind_dir"] for entry in forecast_data]

    # conversie
    timestamps = [datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S") for timestamp in timestamps]

    # verschil in minuten
    timestamps_seconds = [(timestamp - timestamps[0]).total_seconds() / 60 for timestamp in timestamps]

    # Spline-interpolatie
    f_windsnelheid = CubicSpline(timestamps_seconds, windsnelheden)
    f_windrichting = CubicSpline(timestamps_seconds, windrichtingen)

    # tijd van nu + 24u
    now = datetime.now()
    target_time = now + timedelta(hours=24)

    # target
    target_timestamp = (target_time - timestamps[0]).total_seconds() / 60

    #  60 seconden simulatie
    simulation_timestamps = np.linspace(0, target_timestamp, 60)

    # Interpoleren
    simulation_windsnelheden = f_windsnelheid(simulation_timestamps)
    simulation_windrichtingen = f_windrichting(simulation_timestamps)

    # geinterpoleerde waarden saven
    np.save("simulation_windsnelheden.npy", simulation_windsnelheden)
    np.save("simulation_windrichtingen.npy", simulation_windrichtingen)

    
    for timestamp, windsnelheid, windrichting in zip(simulation_timestamps, simulation_windsnelheden, simulation_windrichtingen):
        current_timestamp = timestamps[0] + timedelta(seconds=int(timestamp * 60))
        print(f"Timestamp: {current_timestamp}")
        print(f"Windsnelheid: {windsnelheid} m/s")
        print(f"Windrichting: {windrichting} graden")
        


forecast_data = fetch_weather_data()

if forecast_data is not None:
    process_weather_data(forecast_data)

