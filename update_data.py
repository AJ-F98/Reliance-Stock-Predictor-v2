import os
from datetime import datetime

print(f"Daily update started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

os.system("python data_downloader.py")
os.system("python add_technical_features.py")
os.system("python create_features_and_target.py")
os.system("python train_model.py")

with open("last_update.txt", "w") as f:
    f.write(f"Last successful update: {datetime.now()}")

print("Daily pipeline completed + timestamp updated")