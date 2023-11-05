from models import EntropyModel

load_path = "models/DE_v3/model_1"

threshold = 0.8
myde = EntropyModel.load_from_dir(load_path, threshold=threshold)

myde.quantize()

for i in ["salim", "necesn", "komputer", "telefin", "piyanina"]:
    print(myde.predict(i, threshold=threshold, certain=True))
