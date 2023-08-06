from models import DeepEnsemble

load_path = "models/DE_v1"

no_models = 3
threshold = int(round(no_models * 2 / 3) / no_models * 100) / 100
myde = DeepEnsemble.load_from_dir(load_path, no_models=no_models, threshold=threshold)

myde.quantize()
print(myde.predict("gedek"))
