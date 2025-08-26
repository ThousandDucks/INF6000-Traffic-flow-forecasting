import pandas as pd
from pathlib import Path
import holidays

# Read & clean raw PeMS files (keep only Station 400000)

def process_traffic_data(folders, data_path):
    """
    Reads PeMS4 5-min station files, keeps only
    Timestamp, Station, TotalFlow, filters to Station == 400000,
    rounds timestamps to 5 minutes, drops duplicates.
    Returns a sorted DataFrame.
    """
    all_rows = []

    for folder in folders:
        folder_path = data_path / folder
        file_list = sorted(folder_path.glob("d04_text_station_5min_*.txt.gz"))
        print(f"{folder}: {len(file_list)} files found")

        for file_path in file_list:
            try:
                df = pd.read_csv(file_path, compression='gzip', header=None)

                df = df.iloc[:, :12]
                df.columns = [
                    'Timestamp', 'Station', 'District', 'Freeway', 'Direction',
                    'LaneType', 'StationLength', 'Samples', 'AvgOccupancy',
                    'TotalFlow', 'AvgSpeed', 'Observed'
                ]

                df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

                # Keep the columns we need
                df = df[['Timestamp', 'Station', 'TotalFlow']]

                df = df.dropna(subset=['Timestamp', 'Station', 'TotalFlow'])

                # Keep only Station 400000
                df = df[df['Station'] == 400000]

                df['Timestamp'] = df['Timestamp'].dt.round('5min')
                df = df.drop_duplicates(subset=['Station', 'Timestamp'], keep='first')

                df['TotalFlow'] = pd.to_numeric(df['TotalFlow'], errors='coerce')
                df = df.dropna(subset=['TotalFlow'])
                df['TotalFlow'] = df['TotalFlow'].round(0).astype(int)

                if not df.empty:
                    all_rows.append(df)

                print(f"Processed: {file_path.name} ({df.shape[0]} rows)")

            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")

    if not all_rows:
        raise RuntimeError("No traffic files were successfully processed for Station 400000.")

    full_df = pd.concat(all_rows, ignore_index=True)
    full_df = full_df.sort_values(by=['Station', 'Timestamp']).reset_index(drop=True)
    return full_df



# Creating extra features from timestamp

def engineer_time_features_only(df):
    df = df.copy()
    df = df.set_index('Timestamp').sort_index()
    df = df[~df.index.duplicated()]

    df = df.asfreq('5min')
    df['Station'] = 400000
    df['TotalFlow'] = pd.to_numeric(df['TotalFlow'], errors='coerce')
    df['TotalFlow'] = df['TotalFlow'].interpolate(method='time')

    df = df.reset_index().rename(columns={'index': 'Timestamp'})

    # Round TotalFlow
    df['TotalFlow'] = df['TotalFlow'].round(0).astype('Int64')

    # Time features
    df['hour'] = df['Timestamp'].dt.hour
    df['minute'] = df['Timestamp'].dt.minute
    df['weekday'] = df['Timestamp'].dt.weekday
    df['month'] = df['Timestamp'].dt.month

    # US holiday indicator
    us_holidays = holidays.US()
    df['date'] = df['Timestamp'].dt.date
    df['is_holiday'] = df['date'].apply(lambda x: int(x in us_holidays))
    df.drop(columns=['date'], inplace=True)

    # Peak hour indicator
    def is_peak(h, m):
        morning = (7 <= h < 9) or (h == 9 and m == 0)
        evening = (h == 16 and m >= 30) or (h == 17) or (h == 18 and m <= 30)
        return int(morning or evening)

    df['is_peak_hour'] = df.apply(lambda r: is_peak(r['hour'], r['minute']), axis=1)

    df = df.sort_values(by=['Station', 'Timestamp']).reset_index(drop=True)
    return df



# Main
if __name__ == "__main__":
    base_path = Path(__file__).resolve().parent.parent.parent
    data_path = base_path / "Data"

    folders = ["May 2024", "June 2024"]

    output_path = data_path / "traffic_final_dataset.csv"

    print("Processing traffic data")
    traffic_df = process_traffic_data(folders, data_path)

    final_df = engineer_time_features_only(traffic_df)

    print("Saving final dataset")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print("File saved to:", output_path)
