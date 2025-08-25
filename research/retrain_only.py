import json
from train_from_nasa import train_and_export, FEATURES_CSV, OUT_DIR

info = train_and_export(FEATURES_CSV, OUT_DIR)
print(json.dumps(info, indent=2))