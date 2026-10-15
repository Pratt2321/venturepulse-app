import os
import re
import urllib.request
import pandas as pd
import numpy as np

RAW_DATA_PATH = "data/raw/Unicorn_Companies.csv"
PROCESSED_DATA_PATH = "data/processed/cleaned_unicorns.csv"
DATA_URL = "https://raw.githubusercontent.com/Andres1984/Data-Analysis-with-R/master/Unicorn_Companies.csv"

# Global City to Lat/Lon mapping for top startup hubs
CITY_COORDINATES = {
    # US Hubs
    "San Francisco": (37.7749, -122.4194),
    "New York": (40.7128, -74.0060),
    "Palo Alto": (37.4419, -122.1430),
    "Mountain View": (37.3861, -122.0839),
    "San Jose": (37.3382, -121.8863),
    "Sunnyvale": (37.3688, -122.0363),
    "Redwood City": (37.4852, -122.2364),
    "San Mateo": (37.5630, -122.3255),
    "Santa Clara": (37.3541, -121.9552),
    "Menlo Park": (37.4529, -122.1817),
    "Los Angeles": (34.0522, -118.2437),
    "Culver City": (34.0211, -118.3965),
    "Seattle": (47.6062, -122.3321),
    "Bellevue": (47.6101, -122.2015),
    "Boston": (42.3601, -71.0589),
    "Cambridge": (42.3736, -71.1097),
    "Austin": (30.2672, -97.7431),
    "Chicago": (41.8781, -87.6298),
    "Atlanta": (33.7490, -84.3880),
    "Denver": (39.7392, -104.9903),
    "San Diego": (32.7157, -117.1611),
    "Miami": (25.7617, -80.1918),
    "Philadelphia": (39.9526, -75.1652),
    "Salt Lake City": (40.7608, -111.8910),
    "Lehi": (40.3916, -111.8508),
    "Dallas": (32.7767, -96.7970),
    "Washington": (38.9072, -77.0369),

    # China / Asia Hubs
    "Beijing": (39.9042, 116.4074),
    "Shanghai": (31.2304, 121.4737),
    "Shenzhen": (22.5431, 114.0579),
    "Hangzhou": (30.2741, 120.1551),
    "Guangzhou": (23.1291, 113.2644),
    "Chengdu": (30.5728, 104.0668),
    "Hong Kong": (22.3193, 114.1694),
    "Singapore": (1.3521, 103.8198),
    "Tokyo": (35.6762, 139.6503),
    "Seoul": (37.5665, 126.9780),
    "Bengaluru": (12.9716, 77.5946),
    "Bangalore": (12.9716, 77.5946),
    "Mumbai": (19.0760, 72.8777),
    "Gurugram": (28.4595, 77.0266),
    "Gurgaon": (28.4595, 77.0266),
    "New Delhi": (28.6139, 77.2090),
    "Noida": (28.5355, 77.3910),
    "Jakarta": (-6.2088, 106.8456),

    # Europe Hubs
    "London": (51.5074, -0.1278),
    "Paris": (48.8566, 2.3522),
    "Berlin": (52.5200, 13.4050),
    "Munich": (48.1351, 11.5820),
    "Stockholm": (59.3293, 18.0686),
    "Amsterdam": (52.3676, 4.9041),
    "Dublin": (53.3498, -6.2603),
    "Barcelona": (41.3879, 2.1699),
    "Madrid": (40.4168, -3.7038),
    "Tallinn": (59.4370, 24.7535),
    "Helsinki": (60.1699, 24.9384),
    "Zurich": (47.3769, 8.5417),
    "Geneva": (46.2044, 6.1432),
    "Vienna": (48.2082, 16.3738),
    "Brussels": (50.8503, 4.3517),
    "Copenhagen": (55.6761, 12.5683),
    "Oslo": (59.9139, 10.7522),

    # Middle East / LATAM / Oceania
    "Tel Aviv": (32.0853, 34.7818),
    "Herzliya": (32.1663, 34.8433),
    "Jerusalem": (31.7683, 35.2137),
    "Dubai": (25.2048, 55.2708),
    "Sao Paulo": (-23.5505, -46.6333),
    "SÃ£o Paulo": (-23.5505, -46.6333),
    "Buenos Aires": (-34.6037, -58.3816),
    "Mexico City": (19.4326, -99.1332),
    "Bogota": (4.7110, -74.0721),
    "Sydney": (-33.8688, 151.2093),
    "Melbourne": (-37.8136, 144.9631),
    "Toronto": (43.6532, -79.3832),
    "Vancouver": (49.2827, -123.1207),
    "Montreal": (45.5017, -73.5673)
}

COUNTRY_FALLBACK_COORDINATES = {
    "United States": (37.0902, -95.7129),
    "China": (35.8617, 104.1954),
    "India": (20.5937, 78.9629),
    "United Kingdom": (55.3781, -3.4360),
    "Germany": (51.1657, 10.4515),
    "France": (46.2276, 2.2137),
    "Israel": (31.0461, 34.8516),
    "Canada": (56.1304, -106.3468),
    "Brazil": (-14.2350, -51.9253),
    "Australia": (-25.2744, 133.7751),
    "Singapore": (1.3521, 103.8198),
    "South Korea": (35.9078, 127.7669),
    "Netherlands": (52.1326, 5.2913),
    "Sweden": (60.1282, 18.6435),
    "Japan": (36.2048, 138.2529),
    "Switzerland": (46.8182, 8.2275),
    "Spain": (40.4637, -3.7492),
    "Indonesia": (-0.7893, 113.9213),
    "Mexico": (23.6345, -102.5528),
    "Norway": (60.4720, 8.4689),
    "Estonia": (58.5953, 25.0136),
    "Ireland": (53.1424, -7.6921),
    "United Arab Emirates": (23.4241, 53.8478)
}

def parse_currency_to_millions(val_str):
    if pd.isna(val_str) or not isinstance(val_str, str):
        return np.nan
    val_str = val_str.strip().replace("$", "").replace(",", "")
    if val_str.lower() in ["none", "unknown", ""]:
        return np.nan
    try:
        if val_str.endswith("B"):
            return float(val_str[:-1]) * 1000.0
        elif val_str.endswith("M"):
            return float(val_str[:-1])
        elif val_str.endswith("K"):
            return float(val_str[:-1]) / 1000.0
        else:
            return float(val_str)
    except Exception:
        return np.nan

def clean_industry(ind):
    if pd.isna(ind):
        return "Other"
    ind = str(ind).strip()
    # Normalize common variations
    mapping = {
        "Artificial Intelligence": "Artificial Intelligence",
        "Artificial intelligence": "Artificial Intelligence",
        "Fintech": "Fintech",
        "Finttech": "Fintech",
        "Internet software & services": "Internet Software & Services",
        "E-commerce & direct-to-consumer": "E-Commerce & DTC",
        "Supply chain, logistics, & delivery": "Supply Chain & Logistics",
        "Data management & analytics": "Data Management & Analytics",
        "Mobile & telecommunications": "Mobile & Telecom",
        "Auto & transportation": "Auto & Transportation",
        "Consumer & retail": "Consumer & Retail",
        "Health": "HealthTech & Bio",
        "Cybersecurity": "Cybersecurity",
        "Hardware": "Hardware & IoT",
        "Edtech": "EdTech",
        "Travel": "Travel & Hospitality"
    }
    valid_industries = set(mapping.values())
    for key, target in mapping.items():
        if ind.lower() == key.lower():
            return target
    if ind in valid_industries:
        return ind
    return "Other"

def get_hub_category(city, country):
    city_s = str(city).strip()
    country_s = str(country).strip()
    
    bay_area = ["San Francisco", "Palo Alto", "Mountain View", "San Jose", "Sunnyvale", 
                "Redwood City", "San Mateo", "Santa Clara", "Menlo Park", "Oakland", "Berkeley"]
    if city_s in bay_area:
        return "SF Bay Area"
    elif city_s == "New York":
        return "New York City"
    elif country_s == "United States":
        return "Other US Hubs"
    elif city_s in ["Beijing", "Shanghai", "Shenzhen", "Hangzhou"]:
        return "China Tech Hubs"
    elif city_s in ["London", "Dublin", "Cambridge", "Oxford"]:
        return "UK Hubs"
    elif city_s in ["Berlin", "Munich", "Paris", "Amsterdam", "Stockholm"]:
        return "Europe Hubs"
    elif city_s in ["Bengaluru", "Bangalore", "Mumbai", "Gurugram", "New Delhi"]:
        return "India Tech Hubs"
    elif city_s in ["Tel Aviv", "Herzliya", "Jerusalem"]:
        return "Israel Hubs"
    else:
        return "Global Other"

def download_and_clean():
    print(f"1. Downloading raw data from {DATA_URL}...")
    urllib.request.urlretrieve(DATA_URL, RAW_DATA_PATH)
    df = pd.read_csv(RAW_DATA_PATH, encoding="latin-1")
    print(f"Raw shape: {df.shape}")

    # Standardize column names
    df.rename(columns={
        "Valuation ($B)": "valuation_raw",
        "Select Inverstors": "select_investors",
        "Select Investors": "select_investors",
        "Date Joined": "date_joined",
        "Country": "country",
        "City": "city",
        "Industry": "industry",
        "Founded Year": "founded_year",
        "Total Raised": "total_raised_raw",
        "Investors Count": "investors_count",
        "Deal Terms": "deal_terms"
    }, inplace=True)

    # 1. Clean Valuation
    df["valuation_b"] = df["valuation_raw"].astype(str).str.replace("$", "").str.replace(",", "").astype(float)
    df["valuation_m"] = df["valuation_b"] * 1000.0

    # 2. Clean Total Raised
    df["total_raised_m"] = df["total_raised_raw"].apply(parse_currency_to_millions)
    # For missing total raised, impute sector median or overall median
    median_raised = df["total_raised_m"].median()
    df["total_raised_m_imputed"] = df["total_raised_m"].fillna(median_raised)

    # 3. Clean Dates and Years to Unicorn
    df["year_joined"] = pd.to_datetime(df["date_joined"], errors="coerce").dt.year
    df["founded_year"] = pd.to_numeric(df["founded_year"], errors="coerce")
    
    # Calculate years to unicorn
    df["years_to_unicorn"] = df["year_joined"] - df["founded_year"]
    # Handle anomalies (clip between 1 and 25 years)
    median_years = df.loc[(df["years_to_unicorn"] >= 0) & (df["years_to_unicorn"] <= 30), "years_to_unicorn"].median()
    df["years_to_unicorn"] = df["years_to_unicorn"].apply(lambda y: y if (pd.notna(y) and 0 <= y <= 35) else median_years)

    # 4. Clean Industry
    df["clean_industry"] = df["industry"].apply(clean_industry)

    # 5. Investor Signals
    investor_col = df["select_investors"].fillna("").astype(str)
    df["has_sequoia"] = investor_col.str.contains("Sequoia", case=False).astype(int)
    df["has_a16z"] = investor_col.str.contains("Andreessen|a16z", case=False).astype(int)
    df["has_softbank"] = investor_col.str.contains("SoftBank|Softbank", case=False).astype(int)
    df["has_tiger"] = investor_col.str.contains("Tiger Global", case=False).astype(int)
    df["has_accel"] = investor_col.str.contains("Accel", case=False).astype(int)
    df["has_tier1_investor"] = (
        (df["has_sequoia"] == 1) | 
        (df["has_a16z"] == 1) | 
        (df["has_softbank"] == 1) | 
        (df["has_tiger"] == 1) | 
        (df["has_accel"] == 1)
    ).astype(int)

    # 6. Deal terms & Investors count
    df["investors_count"] = df["investors_count"].fillna(df["investors_count"].median())
    df["deal_terms"] = df["deal_terms"].fillna(df["deal_terms"].median())

    # 7. Hub Categories
    df["hub_category"] = df.apply(lambda r: get_hub_category(r["city"], r["country"]), axis=1)

    # 8. Geocoding
    lats = []
    lons = []
    for _, row in df.iterrows():
        city = str(row["city"]).strip()
        country = str(row["country"]).strip()
        if city in CITY_COORDINATES:
            coords = CITY_COORDINATES[city]
        elif country in COUNTRY_FALLBACK_COORDINATES:
            # Add slight jitter so multiple companies in same country don't overlap completely
            base_lat, base_lon = COUNTRY_FALLBACK_COORDINATES[country]
            jitter_lat = base_lat + np.random.uniform(-0.15, 0.15)
            jitter_lon = base_lon + np.random.uniform(-0.15, 0.15)
            coords = (jitter_lat, jitter_lon)
        else:
            coords = (20.0 + np.random.uniform(-5, 5), 0.0 + np.random.uniform(-5, 5))
        lats.append(coords[0])
        lons.append(coords[1])
        
    df["latitude"] = lats
    df["longitude"] = lons

    # 9. Capital Efficiency Ratio = Valuation ($M) / (Total Raised ($M))
    df["capital_efficiency_ratio"] = (df["valuation_m"] / (df["total_raised_m_imputed"] + 1.0)).round(2)

    # 10. Select final clean columns
    cols = [
        "Company", "valuation_b", "valuation_m", "total_raised_m", "total_raised_m_imputed",
        "clean_industry", "country", "city", "hub_category", "latitude", "longitude",
        "founded_year", "year_joined", "years_to_unicorn", "investors_count", "deal_terms",
        "has_tier1_investor", "has_sequoia", "has_a16z", "has_softbank", "has_tiger",
        "select_investors", "capital_efficiency_ratio"
    ]
    df_clean = df[cols].copy()
    df_clean.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Cleaned dataset saved to {PROCESSED_DATA_PATH}!")
    print(f"Processed shape: {df_clean.shape}")
    print("\nProcessed Summary:")
    print(df_clean.describe())

if __name__ == "__main__":
    download_and_clean()
