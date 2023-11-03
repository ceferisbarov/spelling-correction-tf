from models import DeepEnsemble

load_path = "models/DE_v3"

no_models = 8
threshold = int(round(no_models * 2 / 3) / no_models * 100) / 100
myde = DeepEnsemble.load_from_dir(load_path, no_models=no_models, threshold=threshold)

myde.quantize()

for i in ["salim", "necesn", "komputer", "telefin", "piyanina"]:
    print(myde.predict(i, no_models=no_models, treshold=threshold))
