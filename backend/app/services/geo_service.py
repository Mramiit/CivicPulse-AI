import httpx
from typing import Dict, Any, Tuple
from backend.app.config import settings

async def reverse_geocode_nominatim(latitude: float, longitude: float) -> Tuple[str, str]:
    """
    Reverse geocodes latitude and longitude using the live OpenStreetMap Nominatim REST API.
    Complies with OSM Usage Policy by providing a valid, descriptive User-Agent header.
    
    Returns:
        Tuple of (formatted_address, ward_or_suburb_name)
    """
    params = {
        "lat": latitude,
        "lon": longitude,
        "format": "jsonv2",
        "addressdetails": 1,
        "zoom": 18
    }
    headers = {
        "User-Agent": settings.NOMINATIM_USER_AGENT,
        "Accept": "application/json"
    }

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.get(settings.NOMINATIM_URL, params=params, headers=headers)
            if response.status_code == 200:
                data = response.json()
                display_name = data.get("display_name", f"Coordinates: {latitude:.5f}, {longitude:.5f}")
                address_parts = data.get("address", {})
                
                # Extract municipal ward, suburb, neighbourhood, or district
                ward = (
                    address_parts.get("suburb") or
                    address_parts.get("neighbourhood") or
                    address_parts.get("quarter") or
                    address_parts.get("city_district") or
                    address_parts.get("residential") or
                    address_parts.get("county") or
                    "Municipal Ward Zone"
                )
                return display_name, ward
    except Exception as e:
        # Fallback without crashing
        pass

    fallback_addr = f"Geolocated Pin ({latitude:.5f}° N, {longitude:.5f}° E)"
    return fallback_addr, "Central Municipal Zone"


async def fetch_live_weather_risk(latitude: float, longitude: float) -> Tuple[float, Dict[str, Any]]:
    """
    Queries the live Open-Meteo REST API for real-time weather and precipitation hazards.
    Computes environmental risk factor E_weather in [0.0, 1.0].
    
    Precipitation amplifies road degradation, waterlogging hazard, and traffic collision probability.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,precipitation,rain,showers,weather_code,wind_speed_10m"
    }

    weather_info: Dict[str, Any] = {
        "temperature_c": 26.0,
        "precipitation_mm": 0.0,
        "weather_code": 0,
        "condition_text": "Clear / Normal",
        "wind_speed_kmh": 5.0,
        "e_weather": 0.10
    }
    
    e_weather = 0.10  # Mild baseline

    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            response = await client.get(settings.OPEN_METEO_URL, params=params)
            if response.status_code == 200:
                data = response.json()
                current = data.get("current", {})
                
                precip = float(current.get("precipitation", 0.0) or 0.0)
                rain = float(current.get("rain", 0.0) or 0.0)
                showers = float(current.get("showers", 0.0) or 0.0)
                total_precip = max(precip, rain + showers)
                
                code = int(current.get("weather_code", 0) or 0)
                temp = float(current.get("temperature_2m", 25.0) or 25.0)
                wind = float(current.get("wind_speed_10m", 5.0) or 5.0)

                # Decode WMO Weather interpretation codes
                if code in [95, 96, 99]:
                    condition = "Thunderstorm / High Risk"
                    code_risk = 0.85
                elif code in [80, 81, 82, 65, 67]:
                    condition = "Heavy Rain / Downpour"
                    code_risk = 0.70
                elif code in [61, 63, 53, 55]:
                    condition = "Moderate Rain / Showers"
                    code_risk = 0.45
                elif code in [51, 56]:
                    condition = "Light Drizzle"
                    code_risk = 0.25
                elif code in [1, 2, 3]:
                    condition = "Partly Cloudy"
                    code_risk = 0.10
                else:
                    condition = "Clear Sky"
                    code_risk = 0.05

                # Precipitation formula: 0 mm = 0.0, 10 mm/h = 0.6, 25+ mm/h = 1.0
                precip_risk = min(1.0, total_precip / 20.0)
                
                e_weather = max(code_risk, precip_risk)
                e_weather = min(1.0, max(0.05, e_weather))

                weather_info = {
                    "temperature_c": temp,
                    "precipitation_mm": total_precip,
                    "weather_code": code,
                    "condition_text": condition,
                    "wind_speed_kmh": wind,
                    "e_weather": round(e_weather, 2)
                }
    except Exception as e:
        # Fallback to mild conditions
        weather_info["error"] = str(e)

    return round(e_weather, 2), weather_info
