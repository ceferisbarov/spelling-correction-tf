from models import DeepEnsemble

load_path = "models/DE_v1"
myde = DeepEnsemble.load_from_dir(load_path)

print(myde.predict("gedek"))
