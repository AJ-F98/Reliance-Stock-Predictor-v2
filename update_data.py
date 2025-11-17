# update_data.py
# Run this once per day (e.g., via Windows Task Scheduler or GitHub Actions)
import os
os.system("python data_downloader.py")
os.system("python add_technical_features.py")
os.system("python create_features_and_target.py")
print("Daily data pipeline completed.")