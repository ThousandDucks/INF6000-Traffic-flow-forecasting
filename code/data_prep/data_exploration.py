import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Paths

base_path = Path(__file__).resolve().parent.parent.parent
input_path = base_path / "Data" / "traffic_final_dataset.csv"
stats_output_path = base_path / "Data" / "summary_statistics_no_weather.csv"


# Load dataset

df = pd.read_csv(input_path, parse_dates=['Timestamp'])
print(f"Dataset shape: {df.shape}")


# 1. Summary statistics

summary_cols = [
    'TotalFlow', 'hour', 'minute', 'weekday', 'month',
    'is_peak_hour', 'is_holiday'
]
summary_cols = [c for c in summary_cols if c in df.columns]

summary_stats = df[summary_cols].describe().transpose()
summary_stats.to_csv(stats_output_path)
print("\nSummary statistics saved to:", stats_output_path)
print(summary_stats)


# Distribution of TotalFlow

plt.figure(figsize=(8, 5))
sns.histplot(df['TotalFlow'], bins=50, kde=True)
plt.title("Distribution of Traffic Flow")
plt.xlabel("Total Flow (vehicles / 5 min)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


# Time series plot for station

sample_stations = df['Station'].unique()[:3]
plt.figure(figsize=(12, 6))
for station in sample_stations:
    subset = df[df['Station'] == station].set_index('Timestamp')
    subset['TotalFlow'].plot(label=f'Station {station}')
plt.legend()
plt.title("Traffic Flow Time Series for Sample Stations")
plt.xlabel("Time")
plt.ylabel("Total Flow (vehicles / 5 min)")
plt.tight_layout()
plt.show()


# Correlation heatmap 

for bcol in ['is_peak_hour', 'is_holiday']:
    if bcol in df.columns:
        df[bcol] = df[bcol].astype(int)

if 'month' not in df.columns:
    df['month'] = df['Timestamp'].dt.month

if 'hour' not in df.columns:
    df['hour'] = df['Timestamp'].dt.hour
if 'minute' not in df.columns:
    df['minute'] = df['Timestamp'].dt.minute
if 'weekday' not in df.columns:
    df['weekday'] = df['Timestamp'].dt.weekday

num_cols = [
    'TotalFlow', 'hour', 'minute', 'weekday', 'month',
    'is_peak_hour', 'is_holiday'
]
num_cols = [c for c in num_cols if c in df.columns]  

plt.figure(figsize=(10, 8))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# Boxplot of TotalFlow by Hour

plt.figure(figsize=(12, 6))
sns.boxplot(x='hour', y='TotalFlow', data=df)
plt.title('Traffic Flow by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Total Flow (vehicles / 5 min)')
plt.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()


# Boxplot of TotalFlow by Weekday (with weekday names)

weekday_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
if 'weekday' in df.columns:
    df['weekday_name'] = df['weekday'].map(dict(enumerate(weekday_labels)))
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='weekday_name', y='TotalFlow', data=df, order=weekday_labels)
    plt.title("Traffic Flow by Weekday")
    plt.xlabel("Weekday")
    plt.ylabel("Total Flow (vehicles / 5 min)")
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

print("\nMissing values per column:")
print(df.isna().sum())

# Checking if holiday dates are correct
if 'is_holiday' in df.columns:
    holiday_mask = df['is_holiday'] == 1
    holiday_dates = df.loc[holiday_mask, 'Timestamp'].dt.date.unique()
    print(f"\nUnique holiday dates in dataset ({len(holiday_dates)}):")
    for date in sorted(holiday_dates):
        print(date)

    holiday_counts = df.loc[holiday_mask].groupby(df['Timestamp'].dt.date).size()
    print("\nNumber of rows per holiday date:")
    print(holiday_counts)
