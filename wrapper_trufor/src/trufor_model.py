import os
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class TruForModel:
    def __init__(
        self,
        config,
        model_path: Union[str, Path],
        device: Optional[str] = None,
        save_noiseprint: bool = False,
    ):
        self.config = config
        self.model_path = Path(model_path)
        self.save_noiseprint = save_noiseprint
        self.device = self._resolve_device(device)

        self.model = self._load_model()
        self.model.eval()

    def _resolve_device(self, device: Optional[str]) -> torch.device:
        if device is not None:
            return torch.device(device)

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self):
        if self.config.MODEL.NAME != "detconfcmx":
            raise NotImplementedError(f"Model not implemented: {self.config.MODEL.NAME}")

        from models.cmx.builder_np_conf import myEncoderDecoder as TruForNetwork

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False,
        )

        model = TruForNetwork(cfg=self.config)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(self.device)

        return model

    @staticmethod
    def preprocess(image_path: Union[str, Path]) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        tensor = torch.tensor(
            image.transpose(2, 0, 1),
            dtype=torch.float32,
        ) / 256.0

        return tensor.unsqueeze(0)

    @staticmethod
    def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=np.float32)

        if image.max() > 1.0 or image.min() < 0.0:
            image = image - image.min()
            denom = image.max() if image.max() != 0 else 1.0
            image = image / denom

        return (image * 255.0).clip(0, 255).astype(np.uint8)

    @torch.no_grad()
    def predict(self, image_path: Union[str, Path]) -> Dict[str, np.ndarray]:
        image_path = Path(image_path)

        rgb = self.preprocess(image_path).to(self.device)

        pred, conf, det, noiseprint = self.model(rgb)

        pred = torch.squeeze(pred, 0)
        pred = F.softmax(pred, dim=0)[1]
        pred = pred.cpu().numpy()

        result = {
            "map": pred,
            "imgsize": tuple(rgb.shape[2:]),
        }

        if det is not None:
            result["score"] = torch.sigmoid(det).item()

        if conf is not None:
            conf = torch.squeeze(conf, 0)
            conf = torch.sigmoid(conf)[0]
            result["conf"] = conf.cpu().numpy()

        if self.save_noiseprint and noiseprint is not None:
            noiseprint = torch.squeeze(noiseprint, 0)[0]
            result["noiseprint"] = noiseprint.cpu().numpy()

        return result

    def save_prediction(
        self,
        image_path: Union[str, Path],
        output_path: Union[str, Path],
    ) -> Dict[str, np.ndarray]:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result = self.predict(image_path)

        pred_image = self.normalize_to_uint8(result["map"])
        Image.fromarray(pred_image, mode="L").save(output_path)

        if self.save_noiseprint and "noiseprint" in result:
            noiseprint_path = output_path.with_name(output_path.stem + "_np.png")
            noiseprint_image = self.normalize_to_uint8(result["noiseprint"])
            Image.fromarray(noiseprint_image, mode="L").save(noiseprint_path)

        return result

    def predict_image(
        self,
        image_path: Union[str, Path],
        output_path: Union[str, Path],
    ) -> Dict[str, np.ndarray]:
        return self.save_prediction(image_path, output_path)

    def predict_dir(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"),
    ) -> List[Dict[str, np.ndarray]]:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        image_paths = [
            path for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        ]

        results = []

        for image_path in image_paths:
            relative_path = image_path.relative_to(input_dir)
            output_path = output_dir / relative_path.with_suffix(".png")

            result = self.save_prediction(image_path, output_path)
            results.append(result)

        return results