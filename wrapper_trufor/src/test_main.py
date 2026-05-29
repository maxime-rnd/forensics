import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import _C as config
from config import update_config
from trufor_model import TruForModel

update_config(config, None)

model = TruForModel(
    config=config,
    model_path="D:/Coding/Project/forensics_test/weights/trufor.pth.tar",
    device="cuda:0",  # or "cpu"
    save_noiseprint=True,
)

model.predict_image(
    image_path="D:/Coding/Project/forensics_test/images/tampered1.png",
    output_path="D:/Coding/Project/forensics_test/outputs/tampered1.png",
)

model.predict_dir(
    input_dir="D:/Coding/Project/forensics_test/images",
    output_dir="D:/Coding/Project/forensics_test/outputs",
)