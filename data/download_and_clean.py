import os
import re
import pandas as pd
import numpy as np

RAW_DATA_PATH = "data/raw/cbinsights_unicorns.csv"
PROCESSED_DATA_PATH = "data/processed/cleaned_unicorns.csv"

# Global City coordinates mapping for top tech hubs
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
    "Oakland": (37.8044, -122.2712),
    "Berkeley": (37.8715, -122.2730),
    "Los Angeles": (34.0522, -118.2437),
    "Culver City": (34.0211, -118.3965),
    "Santa Monica": (34.0195, -118.4912),
    "Seattle": (47.6062, -122.3321),
    "Bellevue": (47.6101, -122.2015),
    "Boston": (42.3601, -71.0589),
    "Cambridge": (42.3736, -71.1097),
    "Austin": (30.2672, -97.7431),
    "Chicago": (41.8781, -87.6298),
    "Atlanta": (33.7490, -84.3880),
    "Denver": (39.7392, -104.9903),
    "Boulder": (40.0150, -105.2705),
    "San Diego": (32.7157, -117.1611),
    "Miami": (25.7617, -80.1918),
    "Philadelphia": (39.9526, -75.1652),
    "Salt Lake City": (40.7608, -111.8910),
    "Lehi": (40.3916, -111.8508),
    "Dallas": (32.7767, -96.7970),
    "Plano": (33.0198, -96.6989),
    "Washington": (38.9072, -77.0369),
    "Pittsburgh": (40.4406, -79.9959),

    # China / Asia Hubs
    "Beijing": (39.9042, 116.4074),
    "Shanghai": (31.2304, 121.4737),
    "Shenzhen": (22.5431, 114.0579),
    "Hangzhou": (30.2741, 120.1551),
    "Guangzhou": (23.1291, 113.2644),
    "Chengdu": (30.5728, 104.0668),
    "Wuhan": (30.5928, 114.3055),
    "Nanjing": (32.0603, 118.7969),
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
    "Pune": (18.5204, 73.8567),
    "Hyderabad": (17.3850, 78.4867),
    "Chennai": (13.0827, 80.2707),
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
    "Milan": (45.4642, 9.1900),
    "Rome": (41.9028, 12.4964),

    # Middle East / LATAM / Oceania / Canada
    "Tel Aviv": (32.0853, 34.7818),
    "Herzliya": (32.1663, 34.8433),
    "Jerusalem": (31.7683, 35.2137),
    "Dubai": (25.2048, 55.2708),
    "Riyadh": (24.7136, 46.6753),
    "Sao Paulo": (-23.5505, -46.6333),
    "SÃ£o Paulo": (-23.5505, -46.6333),
    "Buenos Aires": (-34.6037, -58.3816),
    "Mexico City": (19.4326, -99.1332),
    "Bogota": (4.7110, -74.0721),
    "Santiago": (-33.4489, -70.6693),
    "Sydney": (-33.8688, 151.2093),
    "Melbourne": (-37.8136, 144.9631),
    "Brisbane": (-27.4698, 153.0251),
    "Toronto": (43.6532, -79.3832),
    "Vancouver": (49.2827, -123.1207),
    "Montreal": (45.5017, -73.5673),
    "Waterloo": (43.4643, -80.5204)
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
    "United Arab Emirates": (23.4241, 53.8478),
    "Finland": (61.9241, 25.7482),
    "Denmark": (56.2639, 9.5018),
    "Belgium": (50.5039, 4.4699),
    "Austria": (47.5162, 14.5501),
    "Italy": (41.8719, 12.5674),
    "Colombia": (4.5709, -74.2973),
    "Chile": (-35.6751, -71.5430),
    "South Africa": (-30.5595, 22.9375),
    "Vietnam": (14.0583, 108.2772),
    "Thailand": (15.8700, 100.9925),
    "Philippines": (12.8797, 121.7740),
    "Saudi Arabia": (23.8859, 45.0792)
}

def clean_industry(ind):
    if pd.isna(ind):
        return "Enterprise Tech"
    ind_s = str(ind).strip()
    if ind_s in ["Industrial", "Industrials"]:
        return "Industrials"
    elif ind_s in ["Health", "Healthcare & Life Sciences"]:
        return "Healthcare & Life Sciences"
    elif ind_s == "West Palm Beach":
        return "Enterprise Tech"
    elif ind_s in ["Insurance", "Financial Services"]:
        return "Financial Services"
    elif ind_s in ["Consumer & Retail", "Media & Entertainment", "Enterprise Tech"]:
        return ind_s
    return "Enterprise Tech"

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
    elif city_s in ["Beijing", "Shanghai", "Shenzhen", "Hangzhou", "Guangzhou", "Chengdu", "Wuhan", "Nanjing"] or country_s == "China":
        return "China Tech Hubs"
    elif city_s in ["London", "Dublin", "Cambridge", "Oxford"] or country_s == "United Kingdom":
        return "UK & Ireland"
    elif country_s in ["Germany", "France", "Netherlands", "Sweden", "Switzerland", "Spain", "Estonia", "Finland", "Norway", "Denmark", "Belgium", "Austria", "Italy"]:
        return "Europe Hubs"
    elif country_s == "India" or city_s in ["Bengaluru", "Bangalore", "Mumbai", "Gurugram", "New Delhi", "Noida", "Pune", "Hyderabad", "Chennai"]:
        return "India Tech Hubs"
    elif country_s == "Israel" or city_s in ["Tel Aviv", "Herzliya", "Jerusalem"]:
        return "Israel Hubs"
    else:
        return "Global Other"

AI_FRONTIER_COMPANIES = {
    "Anthropic", "OpenAI", "DeepSeek", "Cognition", "Moonshot AI", "Databricks", 
    "Scale AI", "Mistral AI", "Mistral", "xAI", "Cohere", "Perplexity", "Glean", 
    "Harvey", "Adept", "Figure", "Groq", "Etched", "Stability AI", "ElevenLabs", 
    "Character.ai", "Synthesia", "Runway", "Hugging Face", "Midjourney", "Anyscale",
    "Together AI", "Poolside", "Cerebras Systems", "Shield AI", "Anduril"
}

def clean_and_process():
    print(f"Loading raw dataset from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Raw shape: {df.shape}")

    # 1. Clean Valuation
    df["valuation_b"] = df["Valuation ($B)"].astype(str).str.replace("$", "").str.replace(",", "").astype(float)
    df["valuation_m"] = df["valuation_b"] * 1000.0

    # 2. Clean Dates and Unicorn Tenure
    df["date_joined"] = pd.to_datetime(df["Date Joined"], errors="coerce")
    df["year_joined"] = df["date_joined"].dt.year
    # Impute missing years with median year
    median_year = df["year_joined"].median()
    df["year_joined"] = df["year_joined"].fillna(median_year).astype(int)
    # Calculate years active as unicorn up to current observation horizon (2026)
    df["tenure_years"] = (2026 - df["year_joined"]).clip(lower=0.5)

    # 3. Clean Geography
    df["Country"] = df["Country"].fillna("United States")
    df["City"] = df["City"].fillna("Other / Remote")
    df["country"] = df["Country"].str.strip()
    df["city"] = df["City"].str.strip()
    df["hub_category"] = df.apply(lambda r: get_hub_category(r["city"], r["country"]), axis=1)

    # 4. Clean Industry
    df["clean_industry"] = df["Industry"].apply(clean_industry)

    # 5. Frontier AI / DeepTech Indicator
    def detect_ai_frontier(comp, ind, investors):
        comp_s = str(comp).strip()
        if comp_s in AI_FRONTIER_COMPANIES:
            return 1
        name_lower = comp_s.lower()
        if any(kw in name_lower for kw in ["ai", "robotics", "deep", "neural", "intelligence", "cognitive", "data"]):
            if ind == "Enterprise Tech":
                return 1
        return 0

    df["is_ai_frontier"] = df.apply(lambda r: detect_ai_frontier(r["Company"], r["clean_industry"], r["Select Investors"]), axis=1)

    # 6. Investor Signals & Network Capital
    inv_series = df["Select Investors"].fillna("").astype(str)
    df["investors_count"] = inv_series.apply(lambda x: len([s for s in x.split(",") if s.strip()]) if x else 1)
    
    df["has_sequoia"] = inv_series.str.contains("Sequoia", case=False).astype(int)
    df["has_a16z"] = inv_series.str.contains("Andreessen|a16z", case=False).astype(int)
    df["has_khosla"] = inv_series.str.contains("Khosla", case=False).astype(int)
    df["has_thrive"] = inv_series.str.contains("Thrive", case=False).astype(int)
    df["has_google"] = inv_series.str.contains("Google|GV|capitalG", case=False).astype(int)
    df["has_softbank"] = inv_series.str.contains("SoftBank|Softbank", case=False).astype(int)
    df["has_accel"] = inv_series.str.contains("Accel", case=False).astype(int)
    df["has_tiger"] = inv_series.str.contains("Tiger Global", case=False).astype(int)
    df["has_tier1_investor"] = (
        (df["has_sequoia"] == 1) | 
        (df["has_a16z"] == 1) | 
        (df["has_khosla"] == 1) | 
        (df["has_thrive"] == 1) | 
        (df["has_google"] == 1) | 
        (df["has_softbank"] == 1) | 
        (df["has_accel"] == 1) | 
        (df["has_tiger"] == 1)
    ).astype(int)

    # 7. Coordinates Geocoding
    lats = []
    lons = []
    np.random.seed(42)
    for _, row in df.iterrows():
        c_city = str(row["city"]).strip()
        c_country = str(row["country"]).strip()
        if c_city in CITY_COORDINATES:
            base_lat, base_lon = CITY_COORDINATES[c_city]
            coords = (base_lat + np.random.uniform(-0.04, 0.04), base_lon + np.random.uniform(-0.04, 0.04))
        elif c_country in COUNTRY_FALLBACK_COORDINATES:
            base_lat, base_lon = COUNTRY_FALLBACK_COORDINATES[c_country]
            coords = (base_lat + np.random.uniform(-0.25, 0.25), base_lon + np.random.uniform(-0.25, 0.25))
        else:
            coords = (25.0 + np.random.uniform(-3, 3), 10.0 + np.random.uniform(-3, 3))
        lats.append(round(coords[0], 4))
        lons.append(round(coords[1], 4))

    df["latitude"] = lats
    df["longitude"] = lons

    # 8. Capital Efficiency & Valuation Velocity Metrics
    # Valuation velocity: valuation added per year as unicorn ($B/year)
    df["valuation_velocity_b"] = (df["valuation_b"] / df["tenure_years"]).round(2)
    
    # 9. Clean Select Columns
    final_cols = [
        "Company", "valuation_b", "valuation_m", "clean_industry", "country", "city",
        "hub_category", "latitude", "longitude", "date_joined", "year_joined", "tenure_years",
        "investors_count", "is_ai_frontier", "has_tier1_investor", "has_sequoia", "has_a16z",
        "has_khosla", "has_thrive", "has_google", "has_softbank", "has_accel", "has_tiger",
        "Select Investors", "valuation_velocity_b"
    ]
    df_clean = df[final_cols].rename(columns={"Select Investors": "select_investors"})
    
    df_clean.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed dataset saved successfully to {PROCESSED_DATA_PATH}!")
    print(f"Total processed unicorns: {len(df_clean)}")
    print("\nTop 5 Companies in Cleaned Dataset:")
    print(df_clean[["Company", "valuation_b", "clean_industry", "country", "city", "year_joined", "is_ai_frontier"]].head(5))

if __name__ == "__main__":
    clean_and_process()
