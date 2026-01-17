import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# --- 设置 ---

# 1. 【修改】定义单位成本数据
# 注意：运输成本已改为 $/tCO2/km
data = {
    'capture': {
        2035: {'min': 70, 'max': 400}, 
        2040: {'min': 60, 'max': 410},
        2050: {'min': 50, 'max': 200}
    },
    'transport': { 
        2035: {'min': 0.4, 'max': 0.6},
        2040: {'min': 0.35, 'max': 0.5},
        2050: {'min': 0.3, 'max': 0.45}
    },
    'storage': {
        2035: {'min': 35, 'max': 40},   
        2040: {'min': 30, 'max': 35},
        2050: {'min': 25, 'max': 30}
    }
}

# 2. 项目周期设置
start_year = 2035
end_year = 2049
years = list(range(start_year, end_year + 1))
num_years = len(years)

# 3. 项目规模和距离设置
annual_capture_volume =  193531.7175 # 每年封存量 (吨)
transport_distance_km = 255.7        # 【新增】运输距离 (公里)

# --- 代码主体 ---

# ===== 1. 基础模型：计算项目总成本区间 =====

def interpolate_unit_cost(year, cost_type, data_source):
    """线性插值计算某年某类单位成本（$/tCO2 或 $/tCO2/km）"""
    if year <= 2040:
        y1, y2 = 2035, 2040
        interval = 5
    else:
        y1, y2 = 2040, 2050
        interval = 10
    
    fraction = (year - y1) / interval
    
    if isinstance(data_source[cost_type][y1], dict):
        cost_y1 = data_source[cost_type][y1]
        cost_y2 = data_source[cost_type][y2]
        min_val = cost_y1['min'] + (cost_y2['min'] - cost_y1['min']) * fraction
        max_val = cost_y1['max'] + (cost_y2['max'] - cost_y1['max']) * fraction
        return min_val, max_val
    else:
        val_y1 = data_source[cost_type][y1]
        val_y2 = data_source[cost_type][y2]
        interp_val = val_y1 + (val_y2 - val_y1) * fraction
        return interp_val, interp_val

# 计算每年的总成本区间
annual_total_cost_min = []
annual_total_cost_max = []
for year in years:
    cap_min, cap_max = interpolate_unit_cost(year, 'capture', data)
    store_min, store_max = interpolate_unit_cost(year, 'storage', data)
    # 计算单位距离运输成本
    trans_per_km_min, trans_per_km_max = interpolate_unit_cost(year, 'transport', data)
    
    # 计算总运输成本
    trans_min = trans_per_km_min * transport_distance_km
    trans_max = trans_per_km_max * transport_distance_km

    unit_cost_min = cap_min + trans_min + store_min
    unit_cost_max = cap_max + trans_max + store_max
    
    annual_total_cost_min.append(unit_cost_min * annual_capture_volume)
    annual_total_cost_max.append(unit_cost_max * annual_capture_volume)

total_project_cost_min = sum(annual_total_cost_min)
total_project_cost_max = sum(annual_total_cost_max)

print("===== 基础模型结果 =====")
print(f"设定参数: 年封存量 = {annual_capture_volume/1e6:.2f} 百万吨, 运输距离 = {transport_distance_km} km")
print(f"项目总成本区间: {total_project_cost_min/1e6:.2f} - {total_project_cost_max/1e6:.2f} (百万元)")


# ===== 2. 敏感性分析：蒙特卡洛模拟 =====

def run_monte_carlo_for_total_cost(n_simulations, data_source, distance_km):
    """蒙特卡洛模拟，计算给定距离下的项目总成本分布"""
    total_project_costs = []
    
    for _ in range(n_simulations):
        sampled_data = {}
        for cost_type in data_source.keys():
            sampled_data[cost_type] = {}
            for year_point in [2035, 2040, 2050]:
                min_val = data_source[cost_type][year_point]['min']
                max_val = data_source[cost_type][year_point]['max']
                sampled_data[cost_type][year_point] = np.random.uniform(min_val, max_val)
        
        current_sim_total_cost = 0
        for year in years:
            cap_unit_cost, _ = interpolate_unit_cost(year, 'capture', sampled_data)
            store_unit_cost, _ = interpolate_unit_cost(year, 'storage', sampled_data)
            trans_unit_per_km, _ = interpolate_unit_cost(year, 'transport', sampled_data)
            
            total_transport_cost = trans_unit_per_km * distance_km
            total_unit_cost = cap_unit_cost + total_transport_cost + store_unit_cost
            
            current_sim_total_cost += total_unit_cost * annual_capture_volume
            
        total_project_costs.append(current_sim_total_cost)
    
    return total_project_costs

# 运行模拟
n_sim = 5000
total_project_costs_mc = run_monte_carlo_for_total_cost(n_sim, data, transport_distance_km)

# 分析结果
mean_total_cost = np.mean(total_project_costs_mc)
p5 = np.percentile(total_project_costs_mc, 5)
p95 = np.percentile(total_project_costs_mc, 95)

print("\n===== 蒙特卡洛模拟结果 =====")
print(f"基于 {transport_distance_km} km 距离的模拟:")
print(f"项目总成本均值: {mean_total_cost/1e6:.2f} 百万元")
print(f"90%置信区间: {p5/1e6:.2f} - {p95/1e6:.2f} (百万元)")


# ===== 3. 【新增】距离敏感性分析 =====
print("\n正在运行距离敏感性分析...")
distances_to_analyze = np.linspace(10, 500, 25) # 分析从10km到500km的25个点
mean_costs_by_distance = []
p5_costs_by_distance = []
p95_costs_by_distance = []

# 为每个距离点运行一次简化的蒙特卡洛模拟
n_sim_per_distance = 500 # 为提高速度，每次分析使用较少的模拟次数
for dist in distances_to_analyze:
    costs = run_monte_carlo_for_total_cost(n_sim_per_distance, data, dist)
    mean_costs_by_distance.append(np.mean(costs))
    p5_costs_by_distance.append(np.percentile(costs, 5))
    p95_costs_by_distance.append(np.percentile(costs, 95))
print("距离敏感性分析完成。")

# ===== 4. 可视化结果 =====
plt.figure(figsize=(21, 6)) # 加宽画布以容纳三个图

# 图1: 项目总成本分布 (针对设定的150km)
plt.subplot(1, 3, 1)
total_costs_in_millions = [c / 1e6 for c in total_project_costs_mc]
mean_cost_in_millions = mean_total_cost / 1e6
sns.histplot(total_costs_in_millions, kde=True, bins=40, color='#4a6fe3')
plt.axvline(mean_cost_in_millions, color='#e34a4a', linestyle='dashed', linewidth=1, 
            label=f'Mean: {mean_cost_in_millions:.2f} million USD')
plt.title(f'Total Project Cost Distribution (Distance={transport_distance_km}km)')
plt.xlabel('Total Project Cost (Million USD)')
plt.ylabel('Frequency')
plt.legend()

# 图2: 年度总成本变化趋势 (针对设定的150km)
plt.subplot(1, 3, 2)
annual_min_in_millions = [c / 1e6 for c in annual_total_cost_min]
annual_max_in_millions = [c / 1e6 for c in annual_total_cost_max]
annual_avg_in_millions = [(mi + ma) / 2 for mi, ma in zip(annual_min_in_millions, annual_max_in_millions)]
plt.fill_between(years, annual_min_in_millions, annual_max_in_millions, alpha=0.3, color='#6fc3e3', label='Annual Cost Range')
plt.plot(years, annual_avg_in_millions, '#e36f4a', label='Baseline Annual Cost')
plt.title(f'Annual Total Cost Trend (Distance={transport_distance_km}km)')
plt.xlabel('Year')
plt.ylabel('Annual Total Cost (Million USD)')
plt.legend()

# 【新增】图3: 距离敏感性分析图
plt.subplot(1, 3, 3)
mean_costs_in_millions_dist = [c / 1e6 for c in mean_costs_by_distance]
p5_in_millions_dist = [c / 1e6 for c in p5_costs_by_distance]
p95_in_millions_dist = [c / 1e6 for c in p95_costs_by_distance]

plt.plot(distances_to_analyze, mean_costs_in_millions_dist, '#2a9d8f', label='Average Total Cost')
plt.fill_between(distances_to_analyze, p5_in_millions_dist, p95_in_millions_dist, 
                 color='#2a9d8f', alpha=0.2, label='90% Confidence Interval')
plt.title('Total Project Cost vs. Transportation Distance')
plt.xlabel('Transportation Distance (km)')
plt.ylabel('Total Project Cost (Million USD)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()


plt.tight_layout()
plt.savefig('ccus_total_cost_analysis_with_distance.png', dpi=300)
plt.show()    
