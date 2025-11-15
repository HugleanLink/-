import streamlit as st
import requests
import pandas as pd
import folium
from streamlit_folium import st_folium
import numpy as np
from math import radians, sin, cos, sqrt, atan2, asin, degrees

# ================== 配置区 ==================
keywords_default = '商场,购物中心,餐饮服务,中餐厅,西餐厅,咖啡厅,甜品店,酒店,宾馆,酒吧,KTV,电影院,超市,便利店,写字楼,办公楼,地铁站,公交站'
weights = {
    '餐饮服务': 1.0, '中餐厅': 1.0, '西餐厅': 1.0, '咖啡厅': 0.9, '甜品店': 0.9,
    '商场': 0.9, '购物中心': 0.9, '酒店': 0.7, '宾馆': 0.7,
    '酒吧': 0.8, 'KTV': 0.8, '电影院': 0.8,
    '超市': 0.8, '便利店': 0.7,
    '写字楼': 0.6, '办公楼': 0.6,
    '地铁站': 0.4, '公交站': 0.4
}

max_pages = 40
target_radius_km = 8.0
num_clusters = 1
num_primary_stations_per_circle = 5
secondary_radius_km = 4.0
num_secondary_stations = 6
drone_range_km = 12.0
ring_buffer_km = 1.0
outer_buffer_km = 20.0
preset_filter_radius_km = 30.0

# ================== Streamlit UI ==================

st.title("无人机起降站选址系统")
st.write("输入城市名，选择你的高德 API Key，然后点击运行。")

city = st.text_input("城市名称（如：西安市）", "西安市")
api_key = st.text_input("请输入你的高德 API Key", type="password")

run_button = st.button("开始选址分析")

if not run_button:
    st.stop()

if not api_key:
    st.error("请先输入 API Key")
    st.stop()

# ================== 原始功能代码开始 ==================

def get_city_center(city, api_key):
    url = "https://restapi.amap.com/v3/geocode/geo"
    params = {'address': city, 'key': api_key, 'output': 'json'}
    response = requests.get(url, params=params)
    data = response.json()
    if data['status'] == '1' and data['geocodes']:
        lng, lat = map(float, data['geocodes'][0]['location'].split(','))
        return lat, lng
    return None, None

st.info("正在获取城市中心坐标…")
preset_center_lat, preset_center_lng = get_city_center(city, api_key)
if preset_center_lat is None:
    st.error("无法获取城市中心，请检查城市名或 API Key")
    st.stop()
st.success(f"城市中心坐标：({preset_center_lat}, {preset_center_lng})")

# --- Haversine ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def get_destination_point(lat, lng, distance_km, bearing_deg):
    R = 6371
    lat1, lng1, b = radians(lat), radians(lng), radians(bearing_deg)
    lat2 = asin(sin(lat1)*cos(distance_km/R) + cos(lat1)*sin(distance_km/R)*cos(b))
    lng2 = lng1 + atan2(sin(b)*sin(distance_km/R)*cos(lat1),
                        cos(distance_km/R) - sin(lat1)*sin(lat2))
    return degrees(lat2), degrees(lng2)

# --- 获取POI ---
def get_pois(city, kw, api_key):
    pois = []
    for page in range(1, max_pages + 1):
        url = "https://restapi.amap.com/v3/place/text"
        params = {
            'keywords': kw, 'city': city, 'output': 'json',
            'offset': 20, 'page': page, 'key': api_key
        }
        response = requests.get(url, params=params)
        data = response.json()
        if data['status'] != '1' or not data.get('pois'):
            break
        for poi in data['pois']:
            loc = poi.get('location')
            if loc:
                lng, lat = map(float, loc.split(','))
                pois.append({'lat': lat, 'lng': lng, 'name': poi.get('name', '未知')})
    return pd.DataFrame(pois)

# --- 处理POI ---
st.info("📡 开始抓取 POI 数据…")
keyword_list = [k.strip() for k in keywords_default.split(',')]
all_pois = pd.DataFrame()

for kw in keyword_list:
    st.write(f"正在获取：{kw}")
    df = get_pois(city, kw, api_key)
    if not df.empty:
        df['category'] = kw
        all_pois = pd.concat([all_pois, df], ignore_index=True)

if all_pois.empty:
    st.error("未获取到任何 POI，请检查 API Key。")
    st.stop()

# 去重
all_pois.drop_duplicates(subset=['lat', 'lng', 'name'], inplace=True)

# 权重
all_pois['weight'] = all_pois['category'].map(weights).fillna(0.5)

# 过滤郊区
distances_to_center = [haversine(preset_center_lat, preset_center_lng, r['lat'], r['lng'])
                        for _, r in all_pois.iterrows()]
all_pois = all_pois[np.array(distances_to_center) <= preset_filter_radius_km]

st.success(f"市中心 {preset_filter_radius_km} km 范围内，共获取 POI：{len(all_pois)} 个")

# ================== 聚类（使用 SimpleKMeans） ==================
class SimpleKMeans:
    def __init__(self, n_clusters=1, max_iter=100, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        X = np.array(X)
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(X), self.n_clusters, replace=False)
        centers = X[idx]

        for _ in range(self.max_iter):
            distances = np.sqrt(((X[:, None] - centers[None, :]) ** 2).sum(axis=2))
            labels = np.argmin(distances, axis=1)
            new_centers = np.array([X[labels == i].mean(axis=0)
                                    for i in range(self.n_clusters)])
            if np.linalg.norm(centers - new_centers) < self.tol:
                break
            centers = new_centers

        self.centers = centers
        self.labels_ = labels
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

coords = all_pois[['lat', 'lng']].values
weights_arr = all_pois['weight'].values
weighted_coords = np.repeat(coords, (weights_arr * 10).astype(int) + 1, axis=0)

# 主聚类
kmeans = SimpleKMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(weighted_coords)
cluster_centers = kmeans.centers

circles = []
primary_stations = []
secondary_stations = []

st.info("正在计算繁华区…")

# ======= 完全保留你的原始逻辑 =======
for i, (center_lat, center_lng) in enumerate(cluster_centers):

    distances = [haversine(center_lat, center_lng, r['lat'], r['lng'])
                 for _, r in all_pois.iterrows()]
    valid_pois = all_pois[np.array(distances) <= target_radius_km]
    if len(valid_pois) == 0:
        continue

    actual_radius = min(max([d for d in distances if d <= target_radius_km]),
                        target_radius_km)

    circles.append({
        'center_lat': center_lat,
        'center_lng': center_lng,
        'radius_km': actual_radius,
        'poi_count': len(valid_pois)
    })

    # ========== 一级站 ==========
    ring_min = actual_radius - ring_buffer_km
    ring_max = actual_radius + ring_buffer_km
    ring_pois = all_pois[(np.array(distances) >= ring_min) &
                         (np.array(distances) <= ring_max)]

    if len(ring_pois) < num_primary_stations_per_circle:
        st.warning("一级站 POI 不足，使用均匀分布。")
        angle_step = 360 / num_primary_stations_per_circle
        for j in range(num_primary_stations_per_circle):
            angle = j * angle_step
            lat, lng = get_destination_point(center_lat, center_lng,
                                             actual_radius, angle)
            primary_stations.append({
                'id': f'P{i+1}_{j+1}', 'lat': lat, 'lng': lng,
                'circle_id': i + 1
            })
    else:
        ring_coords = ring_pois[['lat', 'lng']].values
        ring_weights = ring_pois['weight'].values
        weighted = np.repeat(ring_coords, (ring_weights * 10).astype(int) + 1, axis=0)

        km = SimpleKMeans(n_clusters=num_primary_stations_per_circle)
        km.fit(weighted)
        for j, (lat, lng) in enumerate(km.centers):
            primary_stations.append({
                'id': f'P{i+1}_{j+1}', 'lat': lat, 'lng': lng,
                'circle_id': i + 1
            })

    # ========== 二级站（同你原来） ==========
    outer_min = actual_radius
    outer_max = actual_radius + outer_buffer_km
    outer_pois = all_pois[(np.array(distances) > outer_min) &
                          (np.array(distances) <= outer_max)]

    num_secondary_total = num_primary_stations_per_circle * num_secondary_stations

    if len(outer_pois) < num_secondary_total:
        st.warning("二级站 POI 不足，使用随机辐射。")
        for pri in primary_stations[-num_primary_stations_per_circle:]:
            for k in range(num_secondary_stations):
                angle = np.random.uniform(0, 360)
                sec_lat, sec_lng = get_destination_point(pri['lat'], pri['lng'],
                                                         secondary_radius_km, angle)
                dist_center = haversine(center_lat, center_lng, sec_lat, sec_lng)
                dist_pri = haversine(pri['lat'], pri['lng'], sec_lat, sec_lng)
                if dist_pri > drone_range_km:
                    continue
                if dist_center <= actual_radius:
                    continue
                secondary_stations.append({
                    'id': f'S{i+1}_{pri["id"].split("_")[1]}_{k+1}',
                    'lat': sec_lat, 'lng': sec_lng,
                    'primary_id': pri['id']
                })
    else:
        outer_coords = outer_pois[['lat', 'lng']].values
        outer_weights = outer_pois['weight'].values
        weighted = np.repeat(outer_coords, (outer_weights * 10).astype(int) + 1, axis=0)

        km = SimpleKMeans(n_clusters=num_secondary_total)
        km.fit(weighted)
        sec_centers = km.centers

        for idx2, (lat, lng) in enumerate(sec_centers):
            nearest = None
            dmin = 1e9
            for pri in primary_stations[-num_primary_stations_per_circle:]:
                d = haversine(pri['lat'], pri['lng'], lat, lng)
                if d < dmin and d <= drone_range_km:
                    nearest = pri
                    dmin = d
            if nearest:
                secondary_stations.append({
                    'id': f'S{i+1}_{idx2+1}',
                    'lat': lat, 'lng': lng,
                    'primary_id': nearest['id']
                })

# ==================地图绘制==================

st.info("正在绘制地图…")

map_center = [all_pois['lat'].mean(), all_pois['lng'].mean()]
m = folium.Map(
    location=map_center,
    zoom_start=11,
    tiles='https://webrd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scale=1&style=8',
    attr='高德地图'
)

# POI
for _, r in all_pois.iterrows():
    folium.CircleMarker([r['lat'], r['lng']],
                        radius=2, color='gray',
                        fill=True, fill_opacity=0.6).add_to(m)

# 繁华区
for idx, c in enumerate(circles):
    folium.Circle(
        [c['center_lat'], c['center_lng']],
        radius=c['radius_km']*1000,
        color='red', weight=3,
        fill=True, fill_opacity=0.15
    ).add_to(m)

# 一级站
for s in primary_stations:
    folium.Marker([s['lat'], s['lng']],
                  icon=folium.Icon(color='orange')).add_to(m)

# 二级站
for s in secondary_stations:
    folium.Marker([s['lat'], s['lng']],
                  icon=folium.Icon(color='blue')).add_to(m)

# 连接线
for sec in secondary_stations:
    for pri in primary_stations:
        if pri['id'] == sec['primary_id']:
            folium.PolyLine([[pri['lat'], pri['lng']], [sec['lat'], sec['lng']]],
                            color='yellow', weight=4).add_to(m)
            break

# 显示地图
st.success("分析完成！")
st_map = st_folium(m, width=1000, height=700)





