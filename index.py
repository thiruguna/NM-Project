import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import networkx as nx
import matplotlib.pyplot as plt

# 1. Simulated Historical Demand Data
def generate_historical_demand(days=30):
    data = {
        'day': np.arange(1, days + 1),
        'demand': [50 + np.random.randint(-10, 10) + i * 0.5 for i in range(days)]
    }
    return pd.DataFrame(data)

# 2. Forecasting Demand
def forecast_demand(df, forecast_days=7):
    x = df[['day']]
    y = df['demand']
    model = LinearRegression()
    model.fit(x, y)
    future_days = pd.DataFrame({'day': np.arange(len(df) + 1, len(df) + forecast_days + 1)})
    future_demand = model.predict(future_days)
    return future_days, future_demand

# 3. Inventory Recommendation Logic
def recommend_inventory(demand_forecast, safety_stock=10):
    return [max(0, int(d) + safety_stock) for d in demand_forecast]

# 4. Logistics Optimization (Simple Route Planner)
def build_logistics_graph():
    G = nx.Graph()
    edges = [
        ('Warehouse', 'CityA', 5),
        ('Warehouse', 'CityB', 8),
        ('CityA', 'CityC', 4),
        ('CityB', 'CityC', 3)
    ]
    G.add_weighted_edges_from(edges)
    return G

def optimize_route(G, source='Warehouse', target='CityC'):
    path = nx.shortest_path(G, source=source, target=target, weight='weight')
    distance = nx.shortest_path_length(G, source=source, target=target, weight='weight')
    return path, distance

# 5. Visualization
def plot_demand(df, future_days, future_demand):
    plt.figure(figsize=(10, 6))
    plt.plot(df['day'], df['demand'], label='Historical Demand')
    plt.plot(future_days['day'], future_demand, label='Forecasted Demand', linestyle='--')
    plt.xlabel('Day')
    plt.ylabel('Units')
    plt.title('Demand Forecasting')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Main Execution
def main():
    print("Generating historical demand data...")
    df = generate_historical_demand()

    print("Forecasting future demand...")
    future_days, future_demand = forecast_demand(df)

    print("Recommending inventory...")
    inventory = recommend_inventory(future_demand)

    print("Optimizing logistics routes...")
    G = build_logistics_graph()
    route, distance = optimize_route(G)

    print("\n--- Supply Chain AI Report ---")
    print("Forecasted Demand:", future_demand.round(2).tolist())
    print("Recommended Inventory:", inventory)
    print("Optimal Route to CityC:", route)
    print("Total Distance:", distance)

    print("\nGenerating demand forecast plot...")
    plot_demand(df, future_days, future_demand)

if __name__ == "__main__":
    main()
