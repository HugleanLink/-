import requests
import pandas as pd
import folium
import random
import math
import numpy as np
from shapely.geometry import Point, Polygon
from scipy.spatial import ConvexHull

# ========================= 配置部分 =========================
API_KEY = ""
CITY = ""
KEYWORDS = "商场,餐饮服务,地铁站,酒店,超市,写字楼"
MAX_PAGES = 8              # 降低运行时间
GRID_SIZE_KM = 5           # 每格边长5km
LEVEL1_COUNT = (10, 15)    # 一级基站数量范围
LEVEL2_PER_L1 = 5          # 每个一级基站对应5个二级基站
MERGE_DISTANCE_KM = 2      # 二级基站合并距离阈值
MIN_DIST_TO_OTHER_L1_KM = 5  # 二级与其他一级基站的最小安全距离

# ========================= 工具函数 =========================
def fetch_all_pois():
    """从高德API获取POI"""
    all_pois = []
    for kw in KEYWORDS.split(","):
        pois = []
        for page in range(1, MAX_PAGES + 1):
            url = "https://restapi.amap.com/v3/place/text"
            params = {
                "keywords": kw,
                "city": CITY,
                "output": "json",
                "offset": 25,
                "page": page,
                "key": API_KEY
            }
            r = requests.get(url, params=params, timeout=10)
            data = r.json()
            if data.get("status") != "1" or not data.get("pois"):
                break
            for poi in data["pois"]:
                loc = poi.get("location")
                if loc:
                    lng, lat = map(float, loc.split(","))
                    pois.append((lng, lat))
        print(f"  {kw} 获取到 {len(pois)} 条POI")
        all_pois.extend(pois)
    return pd.DataFrame(all_pois, columns=["lng", "lat"])

def generate_grid(min_lng, max_lng, min_lat, max_lat, size_km):
    """生成固定网格"""
    km_per_deg = 111
    step = size_km / km_per_deg
    lngs = np.arange(min_lng, max_lng, step)
    lats = np.arange(min_lat, max_lat, step)
    grid = []
    for i, lng in enumerate(lngs):
        for j, lat in enumerate(lats):
            grid.append({"lng": lng + step / 2, "lat": lat + step / 2})
    return pd.DataFrame(grid)

def calc_density(grid_df, poi_df, radius_km=3):
    """计算每个网格点的POI密度"""
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
    """合并相距过近的二级基站"""
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
        # 聚合成一个中心点
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
    """删除离非所属一级基站过近的二级基站"""
    km_per_deg = 111
    filtered = []
    for _, row in level2_df.iterrows():
        keep = True
        for _, p in level1_sites.iterrows():
            # 如果该一级基站不是父基站之一
            if [p["lng"], p["lat"]] not in row["parents"]:
                d = math.dist((row["lng"], row["lat"]), (p["lng"], p["lat"])) * km_per_deg
                if d < min_dist_km:
                    keep = False
                    break
        if keep:
            filtered.append(row)
    print(f"过滤后二级基站数量: {len(filtered)}（删除 {len(level2_df)-len(filtered)} 个过近点）")
    return pd.DataFrame(filtered)

# ========================= 主逻辑 =========================
print(f"开始获取 {CITY} 的POI数据...")
poi_df = fetch_all_pois()
if poi_df.empty:
    raise RuntimeError("❌ 未抓到任何POI，请检查API Key或城市名。")
print(f"✅ 共获取 {len(poi_df)} 条POI")

# 获取范围边界
min_lng, max_lng = poi_df["lng"].min(), poi_df["lng"].max()
min_lat, max_lat = poi_df["lat"].min(), poi_df["lat"].max()

# 生成网格并计算密度
grid_df = generate_grid(min_lng, max_lng, min_lat, max_lat, GRID_SIZE_KM)
grid_df = calc_density(grid_df, poi_df)

# ========================= 一级基站 =========================
grid_sorted = grid_df.sort_values("density", ascending=False)
num_l1 = random.randint(*LEVEL1_COUNT)
level1_sites = grid_sorted.head(num_l1).copy()
print(f"一级基站数量: {num_l1}")

# 计算凸包（密集区域）
hull = ConvexHull(level1_sites[["lng", "lat"]])
polygon = Polygon(level1_sites.iloc[hull.vertices][["lng", "lat"]].values).buffer(-0.02)

# ========================= 二级基站 =========================
level2_sites = []
km_per_deg = 111
for _, l1 in level1_sites.iterrows():
    for i in range(LEVEL2_PER_L1):
        angle = random.uniform(0, 2 * math.pi)
        dist_deg = random.uniform(5, 10) / km_per_deg
        lng = l1["lng"] + dist_deg * math.cos(angle)
        lat = l1["lat"] + dist_deg * math.sin(angle)
        if not polygon.contains(Point(lng, lat)):
            level2_sites.append({
                "lng": lng,
                "lat": lat,
                "parent_lng": l1["lng"],
                "parent_lat": l1["lat"]
            })
level2_df = pd.DataFrame(level2_sites)
print(f"初始二级基站数量: {len(level2_df)}")

# ========================= 合并二级基站 =========================
merged_level2 = merge_close_points(level2_df, MERGE_DISTANCE_KM)
print(f"合并后二级基站数量: {len(merged_level2)}")

# ========================= 过滤离其他一级基站过近的二级站 =========================
filtered_level2 = filter_near_other_primaries(merged_level2, level1_sites, MIN_DIST_TO_OTHER_L1_KM)

# ========================= 地图绘制 =========================
m = folium.Map(location=[poi_df["lat"].mean(), poi_df["lng"].mean()], zoom_start=11)

# 抽样显示POI（浅灰）
for _, row in poi_df.sample(min(400, len(poi_df))).iterrows():
    folium.CircleMarker(
        [row["lat"], row["lng"]],
        radius=2,
        color="#aaaaaa",
        fill=True,
        fill_opacity=0.3
    ).add_to(m)

# 一级基站（红）
for _, row in level1_sites.iterrows():
    folium.CircleMarker(
        [row["lat"], row["lng"]],
        radius=7,
        color="red",
        fill=True,
        fill_opacity=0.9,
        popup="一级基站"
    ).add_to(m)

# 二级基站（蓝） + 连线（保留所有父连接）
for _, row in filtered_level2.iterrows():
    folium.CircleMarker(
        [row["lat"], row["lng"]],
        radius=5,
        color="blue",
        fill=True,
        fill_opacity=0.8,
        popup="二级基站"
    ).add_to(m)
    for parent in row["parents"]:
        folium.PolyLine(
            [[row["lat"], row["lng"]], [parent[1], parent[0]]],
            color="green",
            weight=1.5,
            opacity=0.6
        ).add_to(m)

# 绘制凸包边界
points = level1_sites.iloc[hull.vertices][["lat", "lng"]].values.tolist()
points.append(points[0])
folium.PolyLine(points, color="red", weight=2.5, opacity=0.7).add_to(m)

# ========================= 保存结果 =========================
html_file = f"{CITY}_无人机基站选址_过滤.html"
m.save(html_file)
print(f"✅ 地图已生成：{html_file}")
def run_ga(city_name, api_key):
    global CITY, API_KEY
    CITY = city_name
    API_KEY = api_key

    # 运行完整流程
    poi_df = fetch_all_pois()
    min_lng, max_lng = poi_df["lng"].min(), poi_df["lng"].max()
    min_lat, max_lat = poi_df["lat"].min(), poi_df["lat"].max()
    grid_df = generate_grid(min_lng, max_lng, min_lat, max_lat, GRID_SIZE_KM)
    grid_df = calc_density(grid_df, poi_df)

    # 一级站
    grid_sorted = grid_df.sort_values("density", ascending=False)
    num_l1 = random.randint(*LEVEL1_COUNT)
    level1_sites = grid_sorted.head(num_l1).copy()

    # 凸包
    hull = ConvexHull(level1_sites[["lng", "lat"]])
    polygon = Polygon(level1_sites.iloc[hull.vertices][["lng", "lat"]].values).buffer(-0.02)

    # 二级站
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

    # 绘图对象返回给 Streamlit
    m = folium.Map(location=[poi_df["lat"].mean(), poi_df["lng"].mean()], zoom_start=11)

    # 显示 POI
    for _, r in poi_df.sample(min(400, len(poi_df))).iterrows():
        folium.CircleMarker([r["lat"], r["lng"]], radius=2,
                            color="#aaaaaa", fill=True, fill_opacity=0.3).add_to(m)

    # 一级站
    for _, r in level1_sites.iterrows():
        folium.CircleMarker([r["lat"], r["lng"]], radius=7,
                            color="red", fill=True, fill_opacity=0.9).add_to(m)

    # 二级站 + 连线
    for _, r in final_l2.iterrows():
        folium.CircleMarker([r["lat"], r["lng"]], radius=5,
                            color="blue", fill=True, fill_opacity=0.8).add_to(m)
        for parent in r["parents"]:
            folium.PolyLine([[r["lat"], r["lng"]], [parent[1], parent[0]]],
                            color="green", weight=1.5, opacity=0.6).add_to(m)

    info = {
        "一级站数量": len(level1_sites),
        "二级站数量": len(final_l2),
        "参数": {
            "GRID_SIZE_KM": GRID_SIZE_KM,
            "LEVEL1_COUNT": LEVEL1_COUNT,
            "LEVEL2_PER_L1": LEVEL2_PER_L1
        }
    }

    return m, info
