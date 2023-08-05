from models import DeepEnsemble

load_path = "models/DE_v1"
myde = DeepEnsemble.load_from_dir(load_path, no_models=3, threshold=0.66)
myde.quantize()
print(myde.predict("gedek"))
