import os
import logging
from typing import Union, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from models.bisenet import BiSeNet

logger = logging.getLogger(__name__)


class FaceParser:
    """
    Simplified Face parsing class for training and inference.
    Returns only segmentation masks with support for tensor inputs.
    """
    
    def __init__(self, 
                 model_name: str = "resnet18", 
                 weight_path: str = "/home/yihao/code/DTLS_v2/datapipe/FaceParser/weights/resnet18.pt",
                 num_classes: int = 19,
                 input_size: Tuple[int, int] = (512, 512),
                 device: Optional[torch.device] = None):
        """
        Initialize the FaceParser.
        
        Args:
            model_name: Name of the backbone model (e.g., "resnet18", "resnet34")
            weight_path: Path to the model weights file
            num_classes: Number of segmentation classes
            input_size: Target size for input images
            device: Device to run inference on. If None, will auto-detect
        """
        self.model_name = model_name
        self.weight_path = weight_path
        self.num_classes = num_classes
        self.input_size = input_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Transform for PIL images only
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        # Load the model
        self.model = self._load_model()
        
        logger.info(f"FaceParser initialized with {model_name} on {self.device}")
    
    def _load_model(self) -> torch.nn.Module:
        """Load and initialize the BiSeNet model."""
        model = BiSeNet(self.num_classes, backbone_name=self.model_name)
        model.to(self.device)
        
        if os.path.exists(self.weight_path):
            model.load_state_dict(torch.load(self.weight_path, map_location=self.device))
            logger.info(f"Model weights loaded from {self.weight_path}")
        else:
            raise ValueError(f"Weights not found from given path ({self.weight_path})")
        
        model.eval()
        return model
    
    def _prepare_pil_image(self, image: Image.Image) -> torch.Tensor:
        """Prepare PIL image for inference."""
        resized_image = image.resize(self.input_size, resample=Image.BILINEAR)
        image_tensor = self.transform(resized_image)
        return image_tensor.unsqueeze(0)
    
    def _prepare_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Prepare tensor for inference. Assumes tensor is already normalized."""
        # If tensor has batch dimension, keep it; otherwise add batch dimension
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Resize if necessary
        if image_tensor.shape[-2:] != self.input_size:
            image_tensor = torch.nn.functional.interpolate(
                image_tensor, 
                size=self.input_size, 
                mode='bilinear', 
                align_corners=False
            )
        
        return image_tensor
    
    @torch.no_grad()
    def parse(self, 
              image: Union[str, Image.Image, torch.Tensor],
              original_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Parse face and return segmentation mask.
        
        Args:
            image: PIL Image, tensor, or path to image file
            original_size: (width, height) to resize mask to. If None and input is tensor,
                         returns mask at model input size
            
        Returns:
            np.ndarray: Segmentation mask
        """
        # Handle different input types
        if isinstance(image, str):
            # File path
            if not os.path.exists(image):
                raise ValueError(f"Image path does not exist: {image}")
            pil_image = Image.open(image).convert("RGB")
            original_size = pil_image.size
            image_batch = self._prepare_pil_image(pil_image).to(self.device)
            
        elif isinstance(image, Image.Image):
            # PIL Image
            original_size = image.size
            image_batch = self._prepare_pil_image(image).to(self.device)
            
        elif isinstance(image, torch.Tensor):
            # Tensor (for training)
            image_batch = self._prepare_tensor(image).to(self.device)
            # If original_size not provided, use model input size
            if original_size is None:
                original_size = self.input_size
                
        else:
            raise ValueError("Image must be a PIL Image, tensor, or valid file path")
        
        # Run inference
        output = self.model(image_batch)[0]  # Use feat_out for inference
        
        # Get predicted mask
        predicted_mask = output.squeeze(0).cpu().numpy().argmax(0)
        
        # Resize mask to original size if needed
        if predicted_mask.shape != (original_size[1], original_size[0]):  # (height, width)
            mask_pil = Image.fromarray(predicted_mask.astype(np.uint8))
            restored_mask = mask_pil.resize(original_size, resample=Image.NEAREST)
            predicted_mask = np.array(restored_mask)
        
        return predicted_mask
    
    def parse_tensor_batch(self, 
                          tensor_batch: torch.Tensor,
                          original_sizes: Optional[list] = None) -> list:
        """
        Parse batch of tensors (useful for training).
        
        Args:
            tensor_batch: Batch of image tensors [B, C, H, W]
            original_sizes: List of (width, height) tuples for each image in batch
            
        Returns:
            list: List of segmentation masks
        """
        masks = []
        batch_size = tensor_batch.shape[0]
        
        for i in range(batch_size):
            single_tensor = tensor_batch[i]  # [C, H, W]
            original_size = original_sizes[i] if original_sizes else None
            mask = self.parse(single_tensor, original_size)
            masks.append(mask)
        
        return masks


# Example usage functions
def example_training_usage():
    """Example of how to use during training with tensors."""
    parser = FaceParser(
        model_name="resnet18",
        weight_path="/home/yihao/code/DTLS_v2/datapipe/FaceParser/weights/resnet18.pt"
    )
    
    # Simulate training batch (normalized tensors)
    batch_size = 4
    dummy_batch = torch.randn(batch_size, 3, 512, 512)  # [B, C, H, W]
    
    # Parse single tensor
    single_tensor = dummy_batch[0]  # [C, H, W]
    mask = parser.parse(single_tensor)
    print(f"Single mask shape: {mask.shape}")
    
    # Parse batch of tensors
    masks = parser.parse_tensor_batch(dummy_batch)
    print(f"Batch masks: {len(masks)} masks, each shape: {masks[0].shape}")
    
    return masks


def example_inference_usage():
    """Example of how to use for inference with PIL images."""
    parser = FaceParser()
    
    # Parse from file path
    mask = parser.parse("/home/yihao/code/DTLS_v2/train_result/2025-06-13_18-12/images/train/00001000_gt.png")
    print(f"Mask from file: {mask.shape}")
    
    # Parse PIL image
    pil_image = Image.open("/home/yihao/code/DTLS_v2/train_result/2025-06-13_18-12/images/train/00001000_gt.png").convert("RGB")
    mask = parser.parse(pil_image)
    print(f"Mask from PIL: {mask.shape}")
    
    return mask


if __name__ == "__main__":
    # Example usage
    try:
        # Training usage with tensors
        training_masks = example_training_usage()
        print(f"Training masks: {len(training_masks)} masks, each shape: {training_masks[0].shape}")

        
        # Inference usage with images
        # inference_mask = example_inference_usage()
        # print(f"Inference mask shape: {inference_mask.shape}")
        
    except Exception as e:
        logger.error(f"Error in example: {e}")