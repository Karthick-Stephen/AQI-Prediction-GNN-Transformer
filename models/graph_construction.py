import numpy as np
import pandas as pd

# Haversine distance

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

# Pearson correlation

def pearson_correlation(x, y):
    return np.corrcoef(x, y)[0, 1]

# Wind influence model

def wind_influence(wind_speed, direction, pollutant_concentration):
    # Simplified model for how wind influences concentration
    influence = wind_speed * np.cos(np.radians(direction)) 
    return influence * pollutant_concentration

# Fuzzy source probabilities

def fuzzy_source_probabilities(sources):
    # Assuming sources is a list of tuples (source_id, likelihood)
    total = sum(likelihood for _, likelihood in sources)
    probabilities = {source_id: likelihood / total for source_id, likelihood in sources}
    return probabilities
