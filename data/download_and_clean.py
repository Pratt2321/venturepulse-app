import os
import re
import pandas as pd
import numpy as np

RAW_DATA_PATH = "data/raw/cbinsights_unicorns.csv"
PROCESSED_DATA_PATH = "data/processed/cleaned_unicorns.csv"

# Comprehensive geocoding mapping for all cities and tech hubs worldwide (City, Country)
CITY_COORDINATES = {
    # US - Bay Area / Silicon Valley
    ("San Francisco", "United States"): (37.7749, -122.4194),
    ("Palo Alto", "United States"): (37.4419, -122.1430),
    ("Mountain View", "United States"): (37.3861, -122.0839),
    ("San Jose", "United States"): (37.3382, -121.8863),
    ("Sunnyvale", "United States"): (37.3688, -122.0363),
    ("Redwood City", "United States"): (37.4852, -122.2364),
    ("San Mateo", "United States"): (37.5630, -122.3255),
    ("Santa Clara", "United States"): (37.3541, -121.9552),
    ("Menlo Park", "United States"): (37.4529, -122.1817),
    ("Oakland", "United States"): (37.8044, -122.2712),
    ("Berkeley", "United States"): (37.8715, -122.2730),
    ("Fremont", "United States"): (37.5485, -121.9886),
    ("Campbell", "United States"): (37.2872, -121.9499),
    ("Burlingame", "United States"): (37.5779, -122.3481),
    ("San Carlos", "United States"): (37.5072, -122.2605),
    ("San Ramon", "United States"): (37.7799, -121.9780),
    ("Pleasanton", "United States"): (37.6624, -121.8747),
    ("South San Francisco", "United States"): (37.6547, -122.4077),
    ("Millbrae", "United States"): (37.5985, -122.3872),
    ("Milpitas", "United States"): (37.4323, -121.8996),
    ("Los Altos", "United States"): (37.3852, -122.1141),
    ("Atherton", "United States"): (37.4613, -122.1977),
    ("Alameda", "United States"): (37.7652, -122.2416),
    ("Brisbane", "United States"): (37.6808, -122.3999),
    ("Santa Cruz", "United States"): (36.9741, -122.0308),
    ("Scotts Valley", "United States"): (37.0511, -122.0147),
    ("Roseville", "United States"): (38.7521, -121.2880),

    # US - Southern California
    ("Los Angeles", "United States"): (34.0522, -118.2437),
    ("Culver City", "United States"): (34.0211, -118.3965),
    ("Santa Monica", "United States"): (34.0195, -118.4912),
    ("Venice", "United States"): (33.9850, -118.4695),
    ("Marina del Rey", "United States"): (33.9803, -118.4517),
    ("Beverly Hills", "United States"): (34.0736, -118.4004),
    ("El Segundo", "United States"): (33.9192, -118.4165),
    ("Redondo Beach", "United States"): (33.8492, -118.3884),
    ("Torrance", "United States"): (33.8358, -118.3406),
    ("Long Beach", "United States"): (33.7701, -118.1937),
    ("Glendale", "United States"): (34.1425, -118.2551),
    ("Covina", "United States"): (34.0900, -117.8903),
    ("Irvine", "United States"): (33.6846, -117.8265),
    ("Huntington Beach", "United States"): (33.6595, -117.9988),
    ("Mission Viejo", "United States"): (33.6000, -117.6719),
    ("San Diego", "United States"): (32.7157, -117.1611),
    ("Carlsbad", "United States"): (33.1581, -117.3506),
    ("Encinitas", "United States"): (33.0370, -117.2919),
    ("Santa Barbara", "United States"): (34.4208, -119.6982),
    ("Goleta", "United States"): (34.4358, -119.8276),

    # US - New York Metro & Tri-State
    ("New York", "United States"): (40.7128, -74.0060),
    ("Brooklyn", "United States"): (40.6782, -73.9442),
    ("Long Island City", "United States"): (40.7447, -73.9485),
    ("Port Washington", "United States"): (40.8257, -73.6982),
    ("Islandia", "United States"): (40.7857, -73.1704),
    ("Jersey City", "United States"): (40.7178, -74.0431),
    ("Hoboken", "United States"): (40.7440, -74.0324),
    ("Fort Lee", "United States"): (40.8509, -73.9701),
    ("Berkeley Heights", "United States"): (40.6834, -74.4432),
    ("Chatham", "United States"): (40.7407, -74.3838),
    ("Princeton", "United States"): (40.3573, -74.6672),

    # US - Northeast & New England
    ("Boston", "United States"): (42.3601, -71.0589),
    ("Cambridge", "United States"): (42.3736, -71.1097),
    ("Somerville", "United States"): (42.3876, -71.0995),
    ("Brookline", "United States"): (42.3318, -71.1212),
    ("Waltham", "United States"): (42.3765, -71.2356),
    ("Woburn", "United States"): (42.4793, -71.1523),
    ("Natick", "United States"): (42.2834, -71.3468),
    ("Framingham", "United States"): (42.2793, -71.4162),
    ("Quincy", "United States"): (42.2529, -71.0023),
    ("Burlington", "United States"): (42.5048, -71.1956),
    ("Westborough", "United States"): (42.2695, -71.6161),
    ("Worcester", "United States"): (42.2626, -71.8023),
    ("Philadelphia", "United States"): (39.9526, -75.1652),
    ("Ambler", "United States"): (40.1545, -75.2213),
    ("Pittsburgh", "United States"): (40.4406, -79.9959),
    ("Wilmington", "United States"): (39.7447, -75.5484),
    ("Dover", "United States"): (39.1582, -75.5244),

    # US - Mid-Atlantic & DC Metro
    ("Washington", "United States"): (38.9072, -77.0369),
    ("Washington DC", "United States"): (38.9072, -77.0369),
    ("DC", "United States"): (38.9072, -77.0369),
    ("Alexandria", "United States"): (38.8048, -77.0469),
    ("Arlington", "United States"): (38.8799, -77.1068),
    ("McLean", "United States"): (38.9339, -77.1773),
    ("Vienna", "United States"): (38.9012, -77.2655),
    ("Herndon", "United States"): (38.9696, -77.3861),
    ("Bethesda", "United States"): (38.9847, -77.0947),
    ("Ellicott City", "United States"): (39.2673, -76.7983),

    # US - Texas
    ("Austin", "United States"): (30.2672, -97.7431),
    ("Bastrop", "United States"): (30.1105, -97.3153),
    ("Cedar Park", "United States"): (30.5052, -97.8203),
    ("Dallas", "United States"): (32.7767, -96.7970),
    ("Plano", "United States"): (33.0198, -96.6989),
    ("Irving", "United States"): (32.8140, -96.9489),
    ("Houston", "United States"): (29.7604, -95.3698),

    # US - Pacific Northwest
    ("Seattle", "United States"): (47.6062, -122.3321),
    ("Bellevue", "United States"): (47.6101, -122.2015),
    ("Redmond", "United States"): (47.6740, -122.1215),
    ("Kirkland", "United States"): (47.6769, -122.2060),
    ("Everett", "United States"): (47.9790, -122.2021),
    ("Bellingham", "United States"): (48.7519, -122.4787),
    ("Portland", "United States"): (45.5152, -122.6784),
    ("Bend", "United States"): (44.0582, -121.3153),
    ("Clackamas", "United States"): (45.4076, -122.5695),

    # US - Mountain / Southwest
    ("Denver", "United States"): (39.7392, -104.9903),
    ("Boulder", "United States"): (40.0150, -105.2705),
    ("Colorado Springs", "United States"): (38.8339, -104.8214),
    ("Centennial", "United States"): (39.5792, -104.8769),
    ("Greenwood Village", "United States"): (39.6172, -104.9508),
    ("Salt Lake City", "United States"): (40.7608, -111.8910),
    ("Lehi", "United States"): (40.3916, -111.8508),
    ("Draper", "United States"): (40.5247, -111.8638),
    ("Lindon", "United States"): (40.3436, -111.7208),
    ("South Jordan", "United States"): (40.5622, -111.9297),
    ("Scottsdale", "United States"): (33.4942, -111.9261),
    ("Tempe", "United States"): (33.4255, -111.9400),
    ("Glendale", "United States"): (33.5387, -112.1860),
    ("Las Vegas", "United States"): (36.1699, -115.1398),
    ("Reno", "United States"): (39.5296, -119.8138),
    ("Sparks", "United States"): (39.5349, -119.7527),
    ("Incline Village", "United States"): (39.2499, -119.9466),
    ("Zephyr Cove", "United States"): (39.0063, -119.9482),

    # US - Midwest
    ("Chicago", "United States"): (41.8781, -87.6298),
    ("Northbrook", "United States"): (42.1275, -87.8289),
    ("Detroit", "United States"): (42.3314, -83.0458),
    ("Novi", "United States"): (42.4806, -83.4755),
    ("Columbus", "United States"): (39.9612, -82.9988),
    ("Westerville", "United States"): (40.1245, -82.9246),
    ("Independence", "United States"): (41.3934, -81.6376),
    ("Eden Prairie", "United States"): (44.8547, -93.4708),
    ("Lincoln", "United States"): (40.8136, -96.7026),
    ("Madison", "United States"): (43.0731, -89.4012),
    ("Leawood", "United States"): (38.9161, -94.6191),

    # US - Southeast
    ("Atlanta", "United States"): (33.7490, -84.3880),
    ("Marietta", "United States"): (33.9526, -84.5499),
    ("Miami", "United States"): (25.7617, -80.1918),
    ("Coral Gables", "United States"): (25.7215, -80.2684),
    ("Boca Raton", "United States"): (26.3683, -80.1289),
    ("Plantation", "United States"): (26.1276, -80.2331),
    ("Orlando", "United States"): (28.5383, -81.3792),
    ("Tampa", "United States"): (27.9506, -82.4572),
    ("Sarasota", "United States"): (27.3364, -82.5307),
    ("Jacksonville", "United States"): (30.3322, -81.6557),
    ("Raleigh", "United States"): (35.7796, -78.6382),
    ("Durham", "United States"): (35.9940, -78.8986),
    ("Cary", "United States"): (35.7915, -78.7811),
    ("Charlotte", "United States"): (35.2271, -80.8431),
    ("Fort Mill", "United States"): (35.0074, -80.9451),
    ("Charleston", "United States"): (32.7765, -79.9311),
    ("Nashville", "United States"): (36.1627, -86.7816),
    ("Louisville", "United States"): (38.2527, -85.7585),

    # US - Hawaii
    ("Honolulu", "United States"): (21.3069, -157.8583),

    # Canada
    ("Toronto", "Canada"): (43.6532, -79.3832),
    ("Etobicoke", "Canada"): (43.6205, -79.5132),
    ("Mississauga", "Canada"): (43.5890, -79.6441),
    ("Waterloo", "Canada"): (43.4643, -80.5204),
    ("Kitchener", "Canada"): (43.4516, -80.4925),
    ("Montreal", "Canada"): (45.5017, -73.5673),
    ("Vancouver", "Canada"): (49.2827, -123.1207),
    ("Burnaby", "Canada"): (49.2488, -122.9805),
    ("Coquitlam", "Canada"): (49.2838, -122.7932),

    # UK & Ireland
    ("London", "United Kingdom"): (51.5074, -0.1278),
    ("Cambridge", "United Kingdom"): (52.2053, 0.1218),
    ("Bristol", "United Kingdom"): (51.4545, -2.5879),
    ("Manchester", "United Kingdom"): (53.4808, -2.2426),
    ("St. Albans", "United Kingdom"): (51.7527, -0.3394),
    ("Brentwood", "United Kingdom"): (51.6212, 0.3045),
    ("Peterborough", "United Kingdom"): (52.5695, -0.2405),
    ("Gloucester", "United Kingdom"): (51.8642, -2.2380),
    ("Crewe", "United Kingdom"): (53.0991, -2.4410),
    ("Cheshire", "United Kingdom"): (53.2326, -2.6103),
    ("Ellon", "United Kingdom"): (57.3653, -2.0722),
    ("Dublin", "Ireland"): (53.3498, -6.2603),
    ("Kilkenny", "Ireland"): (52.6541, -7.2448),
    ("Limerick", "Ireland"): (52.6638, -8.6267),

    # Europe - Nordics & Baltics
    ("Stockholm", "Sweden"): (59.3293, 18.0686),
    ("Tallinn", "Estonia"): (59.4370, 24.7535),
    ("Vilnius", "Lithuania"): (54.6872, 25.2797),
    ("Helsinki", "Finland"): (60.1699, 24.9384),
    ("Espoo", "Finland"): (60.2055, 24.6559),
    ("Oulu", "Finland"): (65.0121, 25.4651),
    ("Copenhagen", "Denmark"): (55.6761, 12.5683),
    ("Aarhus", "Denmark"): (56.1629, 10.2039),
    ("Herlev", "Denmark"): (55.7236, 12.4402),
    ("Oslo", "Norway"): (59.9139, 10.7522),

    # Europe - Western & Central
    ("Paris", "France"): (48.8566, 2.3522),
    ("Levallois-Perret", "France"): (48.8932, 2.2879),
    ("La Plaine Saint-Denis", "France"): (48.9189, 2.3619),
    ("Saint Mande", "France"): (48.8427, 2.4187),
    ("Evry", "France"): (48.6298, 2.4418),
    ("Croix", "France"): (50.6775, 3.1504),
    ("Roubaix", "France"): (50.6927, 3.1778),
    ("Montpellier", "France"): (43.6108, 3.8767),
    ("Berlin", "Germany"): (52.5200, 13.4050),
    ("Munich", "Germany"): (48.1351, 11.5820),
    ("Munchen", "Germany"): (48.1351, 11.5820),
    ("Hamburg", "Germany"): (53.5511, 9.9937),
    ("Cologne", "Germany"): (50.9375, 6.9603),
    ("Chemnitz", "Germany"): (50.8278, 12.9214),
    ("Duderstadt", "Germany"): (51.5126, 10.2589),
    ("Gauting", "Germany"): (48.0673, 11.3789),
    ("Gilching", "Germany"): (48.1100, 11.2961),
    ("Metzingen", "Germany"): (48.5381, 9.2858),
    ("Ottobrunn", "Germany"): (48.0658, 11.6669),
    ("Amsterdam", "Netherlands"): (52.3676, 4.9041),
    ("Rotterdam", "Netherlands"): (51.9244, 4.4777),
    ("Brussels", "Belgium"): (50.8503, 4.3517),
    ("Ghent", "Belgium"): (51.0543, 3.7174),
    ("Louvain-la-Neuve", "Belgium"): (50.6681, 4.6144),
    ("Zurich", "Switzerland"): (47.3769, 8.5417),
    ("Geneva", "Switzerland"): (46.2044, 6.1432),
    ("Lausanne", "Switzerland"): (46.5197, 6.6323),
    ("Vienna", "Austria"): (48.2082, 16.3738),
    ("Prague", "Czech Republic"): (50.0755, 14.4378),
    ("Budapest", "Hungary"): (47.4979, 19.0402),
    ("Ljubljana", "Slovenia"): (46.0569, 14.5058),
    ("Leudelange", "Luxembourg"): (49.5667, 6.0667),
    ("Vaduz", "Liechtenstein"): (47.1410, 9.5209),

    # Europe - Southern & Mediterranean
    ("Madrid", "Spain"): (40.4168, -3.7038),
    ("Barcelona", "Spain"): (41.3879, 2.1699),
    ("Donostia-San Sebasti\u221a\u00b0n", "Spain"): (43.3183, -1.9812),
    ("Donostia-San Sebastián", "Spain"): (43.3183, -1.9812),
    ("Banyeres de Mariola", "Spain"): (38.7147, -0.6583),
    ("Lisbon", "Portugal"): (38.7223, -9.1393),
    ("Coimbra", "Portugal"): (40.2033, -8.4103),
    ("Linda-a-Velha", "Portugal"): (38.7183, -9.2394),
    ("Milan", "Italy"): (45.4642, 9.1900),
    ("Rome", "Italy"): (41.9028, 12.4964),
    ("Athens", "Greece"): (37.9838, 23.7275),
    ("St. Julians", "Malta"): (35.9186, 14.4900),

    # Middle East & Turkey
    ("Tel Aviv", "Israel"): (32.0853, 34.7818),
    ("Herzliya", "Israel"): (32.1663, 34.8433),
    ("Jerusalem", "Israel"): (31.7683, 35.2137),
    ("Petah Tikva", "Israel"): (32.0840, 34.8878),
    ("Ramat Gan", "Israel"): (32.0823, 34.8107),
    ("Netanya", "Israel"): (32.3215, 34.8532),
    ("Ra'anana", "Israel"): (32.1848, 34.8707),
    ("Yehud", "Israel"): (32.0336, 34.8906),
    ("Dubai", "United Arab Emirates"): (25.2048, 55.2708),
    ("Riyadh", "Saudi Arabia"): (24.7136, 46.6753),
    ("Dhahran", "Saudi Arabia"): (26.2361, 50.0393),
    ("Istanbul", "Turkey"): (41.0082, 28.9784),
    ("Izmir", "Turkey"): (38.4237, 27.1428),

    # Africa
    ("Cairo", "Egypt"): (30.0444, 31.2357),
    ("Lagos", "Nigeria"): (6.5244, 3.3792),
    ("Bryanston", "South Africa"): (-26.0562, 28.0240),
    ("Mahe", "Seychelles"): (-4.6796, 55.4920),
    ("Victoria", "Seychelles"): (-4.6191, 55.4513),

    # China Hubs & Cities
    ("Beijing", "China"): (39.9042, 116.4074),
    ("Chaoyang District", "China"): (39.9219, 116.4431),
    ("Haidian", "China"): (39.9599, 116.2981),
    ("Haidian District", "China"): (39.9599, 116.2981),
    ("Shanghai", "China"): (31.2304, 121.4737),
    ("Pudong", "China"): (31.2215, 121.5444),
    ("Pudong New Area", "China"): (31.2215, 121.5444),
    ("Changning District", "China"): (31.2204, 121.4246),
    ("Xuhui District", "China"): (31.1883, 121.4365),
    ("Yangpu District", "China"): (31.2598, 121.5260),
    ("Shenzhen", "China"): (22.5431, 114.0579),
    ("Guangzhou", "China"): (23.1291, 113.2644),
    ("Hangzhou", "China"): (30.2741, 120.1551),
    ("Haining", "China"): (30.5408, 120.6908),
    ("Suzhou", "China"): (31.2990, 120.5853),
    ("Nanjing", "China"): (32.0603, 118.7969),
    ("Wuxi", "China"): (31.4912, 120.3119),
    ("Changzhou", "China"): (31.8112, 119.9741),
    ("Hefei", "China"): (31.8206, 117.2272),
    ("Wuhan", "China"): (30.5928, 114.3055),
    ("Changsha", "China"): (28.2282, 112.9388),
    ("Chengdu", "China"): (30.5728, 104.0668),
    ("Chongqing", "China"): (29.5630, 106.5516),
    ("Dongguan", "China"): (23.0205, 113.7518),
    ("Zhuhai", "China"): (22.2707, 113.5767),
    ("Qingdao", "China"): (36.0671, 120.3826),
    ("Wenzhou", "China"): (27.9943, 120.6994),
    ("Guiyang", "China"): (26.6470, 106.6302),
    ("Xuancheng", "China"): (30.9407, 118.7588),
    ("Pizhou", "China"): (34.3167, 117.9500),

    # Hong Kong
    ("Hong Kong", "Hong Kong"): (22.3193, 114.1694),
    ("Kowloon", "Hong Kong"): (22.3193, 114.1694),
    ("Central", "Hong Kong"): (22.2819, 114.1581),

    # India
    ("Bengaluru", "India"): (12.9716, 77.5946),
    ("Bangalore", "India"): (12.9716, 77.5946),
    ("Mumbai", "India"): (19.0760, 72.8777),
    ("Maharashtra", "India"): (19.7515, 75.7139),
    ("Pune", "India"): (18.5204, 73.8567),
    ("Gurugram", "India"): (28.4595, 77.0266),
    ("Gurgaon", "India"): (28.4595, 77.0266),
    ("New Delhi", "India"): (28.6139, 77.2090),
    ("Noida", "India"): (28.5355, 77.3910),
    ("Chennai", "India"): (13.0827, 80.2707),
    ("Hyderabad", "India"): (17.3850, 78.4867),
    ("Jaipur", "India"): (26.9124, 75.7873),

    # Other Asia Pacific
    ("Tokyo", "Japan"): (35.6762, 139.6503),
    ("Tsuruoka", "Japan"): (38.7253, 139.8242),
    ("Seoul", "South Korea"): (37.5665, 126.9780),
    ("Gyeonggi-do", "South Korea"): (37.4138, 127.5183),
    ("Singapore", "Singapore"): (1.3521, 103.8198),
    ("Jakarta", "Indonesia"): (-6.2088, 106.8456),
    ("Bandung", "Indonesia"): (-6.9175, 107.6191),
    ("Bangkok", "Thailand"): (13.7563, 100.5018),
    ("Manila", "Philippines"): (14.5995, 120.9842),
    ("Ho Chi Minh City", "Vietnam"): (10.8231, 106.6297),
    ("Selangor", "Malaysia"): (3.0738, 101.5183),
    ("Tashkent", "Uzbekistan"): (41.2995, 69.2401),

    # Australia & New Zealand
    ("Sydney", "Australia"): (-33.8688, 151.2093),
    ("Surry Hills", "Australia"): (-33.8861, 151.2111),
    ("St Leonards", "Australia"): (-33.8233, 151.1925),
    ("Richmond", "Australia"): (-37.8230, 144.9980),
    ("Melbourne", "Australia"): (-37.8136, 144.9631),
    ("Brisbane", "Australia"): (-27.4698, 153.0251),
    ("Auckland", "New Zealand"): (-36.8485, 174.7633),

    # Latin America & Caribbean
    ("Sao Paulo", "Brazil"): (-23.5505, -46.6333),
    ("S\u00e3o Paulo", "Brazil"): (-23.5505, -46.6333),
    ("Barueri", "Brazil"): (-23.5112, -46.8762),
    ("Curitiba", "Brazil"): (-25.4284, -49.2733),
    ("Buenos Aires", "Argentina"): (-34.6037, -58.3816),
    ("Santiago", "Chile"): (-33.4489, -70.6693),
    ("Bogota", "Colombia"): (4.7110, -74.0721),
    ("Quito", "Ecuador"): (-0.1807, -78.4678),
    ("Mexico City", "Mexico"): (19.4326, -99.1332),
    ("Monterrey", "Mexico"): (25.6866, -100.3161),
    ("Juarez", "Mexico"): (31.6904, -106.4245),
    ("Grand Cayman", "Cayman Islands"): (19.3222, -81.2409),
}

# Country fallback coordinates (centered on tech capital/hubs)
COUNTRY_FALLBACK_COORDINATES = {
    "United States": (37.7749, -122.4194),       # SF Bay Area
    "China": (39.9042, 116.4074),                # Beijing
    "India": (12.9716, 77.5946),                 # Bengaluru
    "United Kingdom": (51.5074, -0.1278),        # London
    "Germany": (52.5200, 13.4050),               # Berlin
    "France": (48.8566, 2.3522),                 # Paris
    "Israel": (32.0853, 34.7818),                # Tel Aviv
    "Canada": (43.6532, -79.3832),               # Toronto
    "Brazil": (-23.5505, -46.6333),              # Sao Paulo
    "Australia": (-33.8688, 151.2093),           # Sydney
    "Singapore": (1.3521, 103.8198),
    "South Korea": (37.5665, 126.9780),          # Seoul
    "Netherlands": (52.3676, 4.9041),            # Amsterdam
    "Sweden": (59.3293, 18.0686),                # Stockholm
    "Japan": (35.6762, 139.6503),                # Tokyo
    "Switzerland": (47.3769, 8.5417),            # Zurich
    "Spain": (40.4168, -3.7038),                 # Madrid
    "Indonesia": (-6.2088, 106.8456),            # Jakarta
    "Mexico": (19.4326, -99.1332),               # Mexico City
    "Norway": (59.9139, 10.7522),                # Oslo
    "Estonia": (59.4370, 24.7535),               # Tallinn
    "Ireland": (53.3498, -6.2603),               # Dublin
    "United Arab Emirates": (25.2048, 55.2708),  # Dubai
    "Finland": (60.1699, 24.9384),               # Helsinki
    "Denmark": (55.6761, 12.5683),               # Copenhagen
    "Belgium": (50.8503, 4.3517),                # Brussels
    "Austria": (48.2082, 16.3738),               # Vienna
    "Italy": (45.4642, 9.1900),                  # Milan
    "Colombia": (4.7110, -74.0721),              # Bogota
    "Chile": (-33.4489, -70.6693),               # Santiago
    "South Africa": (-26.2041, 28.0473),         # Johannesburg
    "Vietnam": (10.8231, 106.6297),              # Ho Chi Minh City
    "Thailand": (13.7563, 100.5018),             # Bangkok
    "Philippines": (14.5995, 120.9842),          # Manila
    "Saudi Arabia": (24.7136, 46.6753),          # Riyadh
    "Hong Kong": (22.3193, 114.1694),
    "New Zealand": (-36.8485, 174.7633),         # Auckland
    "Greece": (37.9838, 23.7275),                # Athens
    "Cayman Islands": (19.3222, -81.2409),       # Grand Cayman
    "Lithuania": (54.6872, 25.2797),             # Vilnius
    "Portugal": (38.7223, -9.1393),              # Lisbon
    "Nigeria": (6.5244, 3.3792),                 # Lagos
    "Seychelles": (-4.6796, 55.4920),            # Mahe
    "Czech Republic": (50.0755, 14.4378),        # Prague
    "Hungary": (47.4979, 19.0402),               # Budapest
    "Turkey": (41.0082, 28.9784),                # Istanbul
    "Egypt": (30.0444, 31.2357),                 # Cairo
    "Slovenia": (46.0569, 14.5058),              # Ljubljana
    "Luxembourg": (49.6116, 6.1319),             # Luxembourg City
    "Liechtenstein": (47.1410, 9.5209),          # Vaduz
    "Malta": (35.9186, 14.4900),                 # St. Julians
    "Ecuador": (-0.1807, -78.4678),              # Quito
    "Malaysia": (3.0738, 101.5183),              # Kuala Lumpur / Selangor
    "Uzbekistan": (41.2995, 69.2401),            # Tashkent
    "Argentina": (-34.6037, -58.3816),           # Buenos Aires
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

    bay_area = [
        "San Francisco", "Palo Alto", "Mountain View", "San Jose", "Sunnyvale", 
        "Redwood City", "San Mateo", "Santa Clara", "Menlo Park", "Oakland", "Berkeley",
        "Fremont", "Campbell", "Burlingame", "San Carlos", "San Ramon", "Pleasanton",
        "South San Francisco", "Millbrae", "Milpitas", "Los Altos", "Atherton", "Alameda"
    ]
    if city_s in bay_area:
        return "SF Bay Area"
    elif city_s == "New York" or city_s in ["Brooklyn", "Long Island City", "Jersey City", "Hoboken"]:
        return "New York City"
    elif country_s == "United States":
        return "Other US Hubs"
    elif city_s in ["Beijing", "Shanghai", "Shenzhen", "Hangzhou", "Guangzhou", "Chengdu", "Wuhan", "Nanjing"] or country_s == "China":
        return "China Tech Hubs"
    elif city_s in ["London", "Dublin", "Cambridge", "Oxford", "Bristol", "Manchester"] or country_s in ["United Kingdom", "Ireland"]:
        return "UK & Ireland"
    elif country_s in ["Germany", "France", "Netherlands", "Sweden", "Switzerland", "Spain", "Estonia", "Finland", "Norway", "Denmark", "Belgium", "Austria", "Italy", "Lithuania", "Portugal", "Czech Republic", "Poland", "Hungary", "Luxembourg"]:
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
    # Deduplicate: remove exact copy-paste duplicate rows (Ramp, Invoca) while preserving legitimate homonyms
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Shape after deduplication: {df.shape}")

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

    # 7. Accurate Geocoding
    lats = []
    lons = []
    np.random.seed(42)
    for _, row in df.iterrows():
        c_city = str(row["city"]).strip()
        c_country = str(row["country"]).strip()
        
        # 1. Exact (City, Country) match
        if (c_city, c_country) in CITY_COORDINATES:
            base_lat, base_lon = CITY_COORDINATES[(c_city, c_country)]
            coords = (base_lat + np.random.uniform(-0.02, 0.02), base_lon + np.random.uniform(-0.02, 0.02))
        else:
            # 2. Case-insensitive city search across all known hubs
            found = False
            for (k_city, k_co), (k_lat, k_lon) in CITY_COORDINATES.items():
                if c_city.lower() == k_city.lower():
                    coords = (k_lat + np.random.uniform(-0.02, 0.02), k_lon + np.random.uniform(-0.02, 0.02))
                    found = True
                    break
            
            if not found:
                # 3. Country fallback (placed in tech capital of that country)
                if c_country in COUNTRY_FALLBACK_COORDINATES:
                    base_lat, base_lon = COUNTRY_FALLBACK_COORDINATES[c_country]
                    coords = (base_lat + np.random.uniform(-0.04, 0.04), base_lon + np.random.uniform(-0.04, 0.04))
                elif c_city in COUNTRY_FALLBACK_COORDINATES:
                    base_lat, base_lon = COUNTRY_FALLBACK_COORDINATES[c_city]
                    coords = (base_lat + np.random.uniform(-0.04, 0.04), base_lon + np.random.uniform(-0.04, 0.04))
                else:
                    # Safe global default (SF Bay Area)
                    coords = (37.7749 + np.random.uniform(-0.03, 0.03), -122.4194 + np.random.uniform(-0.03, 0.03))

        lats.append(round(coords[0], 4))
        lons.append(round(coords[1], 4))

    df["latitude"] = lats
    df["longitude"] = lons

    # 8. Capital Efficiency & Valuation Velocity Metrics
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
    print(df_clean[["Company", "valuation_b", "clean_industry", "country", "city", "year_joined", "latitude", "longitude"]].head(5))

if __name__ == "__main__":
    clean_and_process()
