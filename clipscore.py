import torch
from PIL import Image
import numpy as np
import os
from torchvision import transforms
import clip

# set HF_ENDPOINT=https://hf-mirror.com
class Loader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {"default": "ViT-B/32"}),
                "device": (("cuda", "cpu"),),
                "dtype": (("float16", "bfloat16", "float32"),),
            },
        }

    CATEGORY = "Haojihui/ClipScore"
    FUNCTION = "load"
    RETURN_NAMES = ("MODEL", "PROCESSOR")
    RETURN_TYPES = ("PS_MODEL", "PS_PROCESSOR")

    def load(self, model, device, dtype):
        dtype = torch.float32 if device == "cpu" else getattr(torch, dtype)
        model, preprocess = clip.load(model, device=device)
        return (model, preprocess)


class ImageProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PS_MODEL",),
                "processor": ("PS_PROCESSOR",),
                "device": (("cuda", "cpu"),),
                "images": ("IMAGE",),
            },
        }

    CATEGORY = "Haojihui/ClipScore"
    FUNCTION = "process"
    RETURN_TYPES = ("IMAGE_FEATURES",)

    def process(self, model, processor, device, images):
        print(images.shape)
        numpy = images[0].numpy()
        print(numpy.shape)
        imageTensor = transforms.ToTensor()(numpy)
        print("imageTensor ", imageTensor.shape)
        image = transforms.ToPILImage()(imageTensor)

        img = processor(image)
        
        #image = Image.fromarray(numpy)
        features = model.encode_image(img.to(device))
        return (
            features,
        )
    
class RealImageProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PS_MODEL",),
                "processor": ("PS_PROCESSOR",),
                "device": (("cuda", "cpu"),),
                "images": ("IMAGE",),
            },
        }

    CATEGORY = "Haojihui/ClipScore"
    FUNCTION = "process"
    RETURN_TYPES = ("REAL_FEATURES",)

    def process(self, model, processor, device, images):
        print(images.shape)
        numpy = images[0].numpy()
        print(numpy.shape)
        imageTensor = transforms.ToTensor()(numpy)
        print("imageTensor ", imageTensor.shape)
        image = transforms.ToPILImage()(imageTensor)

        img = processor(image)
        
        #image = Image.fromarray(numpy)
        features = model.encode_image(img.to(device))
        return (
            features,
        )
    
class FakeImageProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PS_MODEL",),
                "processor": ("PS_PROCESSOR",),
                "device": (("cuda", "cpu"),),
                "images": ("IMAGE",),
            },
        }

    CATEGORY = "Haojihui/ClipScore"
    FUNCTION = "process"
    RETURN_TYPES = ("FAKE_FEATURES",)

    def process(self, model, processor, device, images):
        print(images.shape)
        numpy = images[0].numpy()
        print(numpy.shape)
        imageTensor = transforms.ToTensor()(numpy)
        print("imageTensor ", imageTensor.shape)
        image = transforms.ToPILImage()(imageTensor)

        img = processor(image)
        
        #image = Image.fromarray(numpy)
        features = model.encode_image(img.to(device))
        return (
            features,
        )


class TextProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PS_MODEL",),
                "device": (("cuda", "cpu"),),
                "text": ("STRING", {"multiline": True}),
            },
        }

    CATEGORY = "Haojihui/ClipScore"
    FUNCTION = "process"
    RETURN_NAMES = ("TEXT_FEATURES")
    RETURN_TYPES = ("PS_TEXT_FEATURES")

    def process(self, model, device, text):
        data = clip.tokenize(text).squeeze()
        features = model.encode_text(data.to(device))

        return (
            features
        )


class ImageScore:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PS_MODEL",),
                "real_features": ("REAL_FEATURES",),
                "fake_features": ("FAKE_FEATURES",),
                "device": (("cuda", "cpu"),),
            },
            "optional": {
                
            },
        }

    CATEGORY = "Haojihui/ClipScore"
    FUNCTION = "imageScore"
    RETURN_NAMES = ("SCORES", "SCORES1")
    RETURN_TYPES = ("STRING", "FLOAT")

    def imageScore(
        self,
        model,
        real_features,
        fake_features,
        device
    ):
        logit_scale = model.logit_scale.exp()

        # normalize features
        real_features = real_features / real_features.norm(dim=1, keepdim=True).to(torch.float32)
        fake_features = fake_features / fake_features.norm(dim=1, keepdim=True).to(torch.float32)
        
        # calculate scores
        # score = logit_scale * real_features @ fake_features.t()
        # score_acc += torch.diag(score).sum()
        score = logit_scale * (fake_features * real_features).sum()
        score_acc += score
        sample_num += 1

        scores = score_acc / sample_num
        scores_str = str(scores)

        return (scores_str, scores)


NODE_CLASS_MAPPINGS = {
    "HaojihuiClipScoreLoader": Loader,
    "HaojihuiClipScoreImageProcessor": ImageProcessor,
    "HaojihuiClipScoreRealImageProcessor": RealImageProcessor,
    "HaojihuiClipScoreFakeImageProcessor": FakeImageProcessor,
    "HaojihuiClipScoreTextProcessor": TextProcessor,
    "HaojihuiClipScoreImageScore": ImageScore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HaojihuiClipScoreLoader": "Loader",
    "HaojihuiClipScoreImageProcessor": "Image Processor",
    "HaojihuiClipScoreRealImageProcessor": "Real Image Processor",
    "HaojihuiClipScoreFakeImageProcessor": "Fake Image Processor",
    "HaojihuiClipScoreTextProcessor": "Text Processor",
    "HaojihuiClipScoreImageScore": "ImageScore",
}
