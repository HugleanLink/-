import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import requests
from sklearn.cluster import KMeans
from math import radians, sin, cos, sqrt, atan2, asin, degrees
import io

# ======================================
# Streamlit 页面设置
# ======================================
st.set_page_config(page_title="选址", layout="wide")
st.title("起降站选址系统")

st.write("请输入城市名称与高德 API Key，然后开始选址分析。")


# ======================================
# 输入项
# ======================================
city = st.text_input("城市名称（例如：武汉市）", value="")
api_key = st.text_input("高德 API Key", value="", type="password")

start_button = st.button("开始选址分析")

# 若未点击按钮，则不执行后续任何逻辑
if not start_button:
    st.stop()

# 若未输入必要数据
if city.strip() == "" or api_key.strip() == "":
    st.error("城市名称和 API Key 不能为空。")
    st.stop()

# --- 按钮 ---
if st.button("开始选址分析"):
    st.session_state["run_analysis"] = True


# --- 只有在 run_analysis=True 才运行下面的全部逻辑 ---
if "run_analysis" not in st.session_state:
    st.stop()

if not st.session_state["run_analysis"]:
    st.stop()

# =====================
# （从这里开始放所有“城市中心 → 获取POI → 聚类 → 绘图 → 下载”逻辑）
# =====================


# ======================================
# 固定参数（与你原始脚本一致）
# ======================================
keywords = '商场,购物中心,餐饮服务,中餐厅,西餐厅,咖啡厅,甜品店,酒店,宾馆,酒吧,KTV,电影院,超市,便利店,写字楼,办公楼,地铁站,公交站'

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


# ======================================
# 工具函数（保持不变）
# ======================================
def get_city_center(city, api_key):
    url = "https://restapi.amap.com/v3/geocode/geo"
    params = {'address': city, 'key': api_key}
    try:
        data = requests.get(url, params=params).json()
        if data['status'] == '1' and data['geocodes']:
            lng, lat = map(float, data['geocodes'][0]['location'].split(','))
            return lat, lng
        return None, None
    except:
        return None, None


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def get_destination_point(lat, lng, distance_km, bearing_deg):
    R = 6371
    lat1 = radians(lat)
    lng1 = radians(lng)
    bearing = radians(bearing_deg)
    lat2 = asin(sin(lat1) * cos(distance_km / R) +
                cos(lat1) * sin(distance_km / R) * cos(bearing))
    lng2 = lng1 + atan2(
        sin(bearing) * sin(distance_km / R) * cos(lat1),
        cos(distance_km / R) - sin(lat1) * sin(lat2)
    )
    return (degrees(lat2), degrees(lng2))


def get_pois(city, kw, api_key, page_size=20, max_pages=40):
    pois = []
    for page in range(1, max_pages + 1):
        url = "https://restapi.amap.com/v3/place/text"
        params = {
            'keywords': kw, 'city': city, 'output': 'json',
            'offset': page_size, 'page': page, 'key': api_key
        }
        try:
            data = requests.get(url, params=params, timeout=10).json()
            if data['status'] != '1' or not data.get('pois'):
                break
            for poi in data['pois']:
                if poi.get('location'):
                    lng, lat = map(float, poi['location'].split(','))
                    pois.append({
                        'lat': lat,
                        'lng': lng,
                        'name': poi['name']
                    })
        except:
            break
    return pd.DataFrame(pois)


# ======================================
# 获取城市中心
# ======================================
st.write("正在获取城市中心坐标")

with st.spinner("联系高德 API 中"):
    preset_center_lat, preset_center_lng = get_city_center(city, api_key)

if preset_center_lat is None:
    st.error("无法获取城市中心，请检查城市名称或 API Key。")
    st.stop()

st.success(f"城市中心位置：({preset_center_lat:.5f}, {preset_center_lng:.5f})")


# ======================================
# 获取 POI 数据
# ======================================
st.write("正在获取 POI 数据")

keyword_list = [k.strip() for k in keywords.split(",")]
all_pois = pd.DataFrame()

for kw in keyword_list:
    st.write(f"获取 `{kw}` 数据…")
    df = get_pois(city, kw, api_key, max_pages=max_pages)
    if not df.empty:
        df["category"] = kw
        all_pois = pd.concat([all_pois, df], ignore_index=True)

if all_pois.empty:
    st.error("POI 数据为空！请检查 API Key 是否有效。")
    st.stop()

all_pois.drop_duplicates(subset=["lat", "lng", "name"], inplace=True)
all_pois["weight"] = all_pois["category"].map(weights).fillna(0.5)

dists = [
    haversine(preset_center_lat, preset_center_lng, r["lat"], r["lng"])
    for _, r in all_pois.iterrows()
]
all_pois = all_pois[np.array(dists) <= preset_filter_radius_km]

st.success(f"有效 POI 数量：{len(all_pois)}")


# ======================================
# 选址聚类逻辑（你的原版全部保留）
# ======================================
coords = all_pois[['lat', 'lng']].values
weights_array = all_pois['weight'].values
weighted_coords = np.repeat(coords, (weights_array * 10).astype(int) + 1, axis=0)

kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(weighted_coords)
cluster_centers = kmeans.cluster_centers_

circles = []
primary_stations = []
secondary_stations = []

# 遍历处理每个圈（与你原版一致）
for i, (center_lat, center_lng) in enumerate(cluster_centers):

    distances = [haversine(center_lat, center_lng, row['lat'], row['lng'])
                 for _, row in all_pois.iterrows()]
    valid_pois = all_pois[np.array(distances) <= target_radius_km]

    if len(valid_pois) == 0:
        continue

    actual_radius = min(max([d for d in distances if d <= target_radius_km]), target_radius_km)

    circles.append({
        'center_lat': center_lat,
        'center_lng': center_lng,
        'radius_km': actual_radius,
        'poi_count': len(valid_pois)
    })

    # 一级站
    ring_min = actual_radius - ring_buffer_km
    ring_max = actual_radius + ring_buffer_km
    ring_pois = all_pois[(np.array(distances) >= ring_min) & (np.array(distances) <= ring_max)]

    if len(ring_pois) < num_primary_stations_per_circle:
        angle_step = 360 / num_primary_stations_per_circle
        for j in range(num_primary_stations_per_circle):
            angle = j * angle_step
            pri_lat, pri_lng = get_destination_point(center_lat, center_lng, actual_radius, angle)
            primary_stations.append({
                'id': f'P{i+1}_{j+1}',
                'lat': pri_lat,
                'lng': pri_lng,
                'circle_id': i+1
            })
    else:
        ring_coords = ring_pois[['lat','lng']].values
        ring_weights = ring_pois['weight'].values
        weighted_ring = np.repeat(ring_coords, (ring_weights * 10).astype(int) + 1, axis=0)

        kmeans_ring = KMeans(n_clusters=num_primary_stations_per_circle, random_state=42, n_init=10)
        centers_ring = kmeans_ring.fit(weighted_ring).cluster_centers_

        for j,(pri_lat,pri_lng) in enumerate(centers_ring):
            primary_stations.append({
                'id': f'P{i+1}_{j+1}',
                'lat': pri_lat,
                'lng': pri_lng,
                'circle_id': i+1
            })

    # 二级站
    outer_min = actual_radius
    outer_max = actual_radius + outer_buffer_km
    outer_pois = all_pois[(np.array(distances) > outer_min) & (np.array(distances) <= outer_max)]

    need_total = num_primary_stations_per_circle * num_secondary_stations

    if len(outer_pois) < need_total:
        for pri in primary_stations[-num_primary_stations_per_circle:]:
            pri_lat, pri_lng = pri['lat'], pri['lng']

            for k in range(num_secondary_stations):
                angle = np.random.uniform(0, 360)
                sec_lat, sec_lng = get_destination_point(pri_lat, pri_lng, secondary_radius_km, angle)

                d1 = haversine(pri_lat, pri_lng, sec_lat, sec_lng)
                d2 = haversine(center_lat, center_lng, sec_lat, sec_lng)

                if d1 > drone_range_km:
                    continue
                if d2 <= actual_radius:
                    continue

                secondary_stations.append({
                    'id': f'S{i+1}_{pri["id"]}_{k+1}',
                    'lat': sec_lat,
                    'lng': sec_lng,
                    'primary_id': pri['id']
                })
    else:
        outer_coords = outer_pois[['lat','lng']].values
        outer_weights = outer_pois['weight'].values
        weighted_outer = np.repeat(outer_coords, (outer_weights * 10).astype(int) + 1, axis=0)

        kmeans_outer = KMeans(n_clusters=need_total, random_state=42, n_init=10)
        centers_outer = kmeans_outer.fit(weighted_outer).cluster_centers_

        idx = 1
        for sec_lat,sec_lng in centers_outer:
            nearest = None
            min_dist = 9999

            for pri in primary_stations[-num_primary_stations_per_circle:]:
                dist = haversine(pri['lat'], pri['lng'], sec_lat, sec_lng)
                if dist < min_dist and dist <= drone_range_km:
                    min_dist = dist
                    nearest = pri

            if nearest:
                secondary_stations.append({
                    'id': f'S{i+1}_{idx}',
                    'lat': sec_lat,
                    'lng': sec_lng,
                    'primary_id': nearest['id']
                })
                idx += 1


# ======================================
# 绘制地图
# ======================================
st.write("选址结果地图")

map_center = [all_pois["lat"].mean(), all_pois["lng"].mean()]
m = folium.Map(
    location=map_center,
    zoom_start=11,
    tiles='https://webrd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scale=1&style=8',
    attr="高德地图"
)

# POI 点
for _, row in all_pois.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lng"]],
        radius=2, color="gray", fill=True, fill_opacity=0.6
    ).add_to(m)

# 繁华区圆圈
for idx, c in enumerate(circles):
    folium.Circle(
        location=[c["center_lat"], c["center_lng"]],
        radius=c["radius_km"] * 1000,
        color="red", weight=3,
        fill=True, fill_color="red", fill_opacity=0.2
    ).add_to(m)

    folium.Marker(
        location=[c["center_lat"], c["center_lng"]],
        popup=f"繁华中心 {idx+1}",
        icon=folium.Icon(color="red", icon="star")
    ).add_to(m)

# 一级站
for st_p in primary_stations:
    folium.Marker(
        location=[st_p["lat"], st_p["lng"]],
        popup=f"一级站 {st_p['id']}",
        icon=folium.Icon(color="orange", icon="star")
    ).add_to(m)

# 二级站
for st_s in secondary_stations:
    folium.Marker(
        location=[st_s["lat"], st_s["lng"]],
        popup=f"二级站 {st_s['id']}（服务 {st_s['primary_id']}）",
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)

# 连线
for sec in secondary_stations:
    for pri in primary_stations:
        if sec["primary_id"] == pri["id"]:
            folium.PolyLine(
                locations=[[pri["lat"], pri["lng"]], [sec["lat"], sec["lng"]]],
                color="yellow", weight=3, opacity=0.7
            ).add_to(m)
            break

# 显示地图
st_folium(m, width=900, height=600)


# ======================================
# 导出 CSV
# ======================================
csv_data = []

for idx, c in enumerate(circles):
    csv_data.append({
        '类型': '圆圈',
        '区编号': idx + 1,
        '中心纬度': round(c['center_lat'], 6),
        '中心经度': round(c['center_lng'], 6),
        '半径_km': round(c['radius_km'], 2),
        'POI数量': c['poi_count']
    })

for station in primary_stations:
    csv_data.append({
        '类型': '一级站',
        '区编号': station['id'],
        '中心纬度': round(station['lat'], 6),
        '中心经度': round(station['lng'], 6),
        '半径_km': '',
        'POI数量': '',
        '服务圈': station['circle_id']
    })

for station in secondary_stations:
    csv_data.append({
        '类型': '二级站',
        '区编号': station['id'],
        '中心纬度': round(station['lat'], 6),
        '中心经度': round(station['lng'], 6),
        '半径_km': '',
        'POI数量': '',
        '服务于': station['primary_id']
    })

csv_df = pd.DataFrame(csv_data)


# 下载 CSV
csv_buf = io.StringIO()
csv_df.to_csv(csv_buf, index=False, encoding="utf-8-sig")


st.write("下载分析结果")
# 下载 HTML
html_str = m.get_root().render()

st.download_button(
    "下载 HTML 地图文件",
    data=html_str,
    file_name=f"{city}_选址地图.html",
    mime="text/html"
)

# 下载 CSV
st.download_button(
    "下载站点数据 CSV",
    data=csv_buf.getvalue(),
    file_name=f"{city}_选址结果.csv",
    mime="text/csv"
)





