import requests
import pandas as pd
import folium
import random
import math
import numpy as np
from shapely.geometry import Point, Polygon
from scipy.spatial import ConvexHull

# ========================= 默认配置 =========================
DEFAULT_KEYWORDS = "商场,餐饮服务,地铁站,酒店,超市,写字楼"
GRID_SIZE_KM = 5
LEVEL1_COUNT = (10, 15)
LEVEL2_PER_L1 = 5
MERGE_DISTANCE_KM = 2
MIN_DIST_TO_OTHER_L1_KM = 5

# 运行时会通过 run_ga() 动态修改
API_KEY = ""
CITY = ""
KEYWORDS = DEFAULT_KEYWORDS

# ========================= 工具函数 =========================

def fetch_all_pois():
    """从高德API获取POI"""
    all_pois = []
    for kw in KEYWORDS.split(","):
        pois = []
        for page in range(1, 9):   # MAX_PAGES = 8
            url = "https://restapi.amap.com/v3/place/text"
            params = {
                "keywords": kw,
                "city": CITY,
                "output": "json",
                "offset": 25,
                "page": page,
                "key": API_KEY
            }
            r = requests.get(url, params=params, timeout=10).json()
            if r.get("status") != "1" or not r.get("pois"):
                break
            for poi in r["pois"]:
                loc = poi.get("location")
                if loc:
                    lng, lat = map(float, loc.split(","))
                    pois.append((lng, lat))
        all_pois.extend(pois)
    return pd.DataFrame(all_pois, columns=["lng", "lat"])


def generate_grid(min_lng, max_lng, min_lat, max_lat, size_km):
    km_per_deg = 111
    step = size_km / km_per_deg
    lngs = np.arange(min_lng, max_lng, step)
    lats = np.arange(min_lat, max_lat, step)
    grid = []
    for lng in lngs:
        for lat in lats:
            grid.append({"lng": lng + step / 2, "lat": lat + step / 2})
    return pd.DataFrame(grid)


def calc_density(grid_df, poi_df, radius_km=3):
    km_per_deg = 111
    r = radius_km / km_per_deg
    densities = []
    for _, g in grid_df.iterrows():
        count = ((abs(poi_df["lng"] - g["lng"]) < r) &
                 (abs(poi_df["lat"] - g["lat"]) < r)).sum()
        densities.append(count)
    grid_df["density"] = densities
    return grid_df


def merge_close_points(df, threshold_km):
    km_per_deg = 111
    threshold_deg = threshold_km / km_per_deg
    merged = []
    used = set()
    for i in range(len(df)):
        if i in used:
            continue
        cluster = [i]
        for j in range(i + 1, len(df)):
            if j in used:
                continue
            d = math.dist((df.iloc[i]["lng"], df.iloc[i]["lat"]),
                          (df.iloc[j]["lng"], df.iloc[j]["lat"]))
            if d < threshold_deg:
                cluster.append(j)
                used.add(j)
        used.add(i)
        cluster_points = df.iloc[cluster]
        lng_mean = cluster_points["lng"].mean()
        lat_mean = cluster_points["lat"].mean()
        parents = cluster_points[["parent_lng", "parent_lat"]].values.tolist()
        merged.append({
            "lng": lng_mean,
            "lat": lat_mean,
            "parents": parents
        })
    return pd.DataFrame(merged)


def filter_near_other_primaries(level2_df, level1_sites, min_dist_km):
    km_per_deg = 111
    filtered = []
    for _, row in level2_df.iterrows():
        keep = True
        for _, p in level1_sites.iterrows():
            if [p["lng"], p["lat"]] not in row["parents"]:
                d = math.dist((row["lng"], row["lat"]), (p["lng"], p["lat"])) * km_per_deg
                if d < min_dist_km:
                    keep = False
                    break
        if keep:
            filtered.append(row)
    return pd.DataFrame(filtered)


# ========================= 改造后的入口函数 =========================

def run_ga(city_name, api_key, keywords=DEFAULT_KEYWORDS):
    """
    city_name: 网站输入的城市
    api_key: 用户输入的高德 API Key
    keywords: 可选，自定义关键词
    """
    global CITY, API_KEY, KEYWORDS
    CITY = city_name
    API_KEY = api_key
    KEYWORDS = keywords

    # 获取 POI
    poi_df = fetch_all_pois()
    if poi_df.empty:
        raise RuntimeError("GA 未获取到任何 POI，请检查城市或 API Key")

    min_lng, max_lng = poi_df["lng"].min(), poi_df["lng"].max()
    min_lat, max_lat = poi_df["lat"].min(), poi_df["lat"].max()

    # 网格密度
    grid_df = generate_grid(min_lng, max_lng, min_lat, max_lat, GRID_SIZE_KM)
    grid_df = calc_density(grid_df, poi_df)

    # 一级基站
    grid_sorted = grid_df.sort_values("density", ascending=False)
    num_l1 = random.randint(*LEVEL1_COUNT)
    level1_sites = grid_sorted.head(num_l1).copy()

    hull = ConvexHull(level1_sites[["lng", "lat"]])
    polygon = Polygon(level1_sites.iloc[hull.vertices][["lng", "lat"]].values).buffer(-0.02)

    # 二级
    level2_sites = []
    km_per_deg = 111
    for _, l1 in level1_sites.iterrows():
        for _ in range(LEVEL2_PER_L1):
            angle = random.uniform(0, 2 * math.pi)
            dist_deg = random.uniform(5, 10) / km_per_deg
            lng = l1["lng"] + dist_deg * math.cos(angle)
            lat = l1["lat"] + dist_deg * math.sin(angle)
            if not polygon.contains(Point(lng, lat)):
                level2_sites.append({
                    "lng": lng, "lat": lat,
                    "parent_lng": l1["lng"], "parent_lat": l1["lat"]
                })

    level2_df = pd.DataFrame(level2_sites)
    merged = merge_close_points(level2_df, MERGE_DISTANCE_KM)
    final_l2 = filter_near_other_primaries(merged, level1_sites, MIN_DIST_TO_OTHER_L1_KM)

    # 绘图
    m = folium.Map(location=[poi_df["lat"].mean(), poi_df["lng"].mean()], zoom_start=11)

    for _, r in poi_df.sample(min(400, len(poi_df))).iterrows():
        folium.CircleMarker([r["lat"], r["lng"]], radius=2,
                            color="#aaaaaa", fill=True, fill_opacity=0.3).add_to(m)

    for _, r in level1_sites.iterrows():
        folium.CircleMarker([r["lat"], r["lng"]], radius=7,
                            color="red", fill=True, fill_opacity=0.9).add_to(m)

    for _, r in final_l2.iterrows():
        folium.CircleMarker([r["lat"], r["lng"]], radius=5,
                            color="blue", fill=True, fill_opacity=0.8).add_to(m)
        for parent in r["parents"]:
            folium.PolyLine([[r["lat"], r["lng"]], [parent[1], parent[0]]],
                            color="green", weight=1.5, opacity=0.6).add_to(m)

    info = {
        "城市": CITY,
        "一级站数量": len(level1_sites),
        "二级站数量": len(final_l2),
    }

    return m, info
