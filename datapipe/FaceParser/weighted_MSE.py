def test_with_face_image_pair(hq_image_path: str, 
                             lq_image_path: str,
                             output_dir: str = "./output",
                             face_parser_model_name: str = "resnet18",
                             face_parser_weight_path: str = "./weights/resnet18.pt",
                             face_weight: float = 2.0,
                             background_weight: float = 1.0,
                             face_class_indices: Union[int, List[int]] = 1,
                             device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Test the FaceParserWeightedMSELoss with HQ and LQ face image pair and save outputs.
    
    Args:
        hq_image_path: Path to high-quality face image
        lq_image_path: Path to low-quality face image
        output_dir: Directory to save outputs
        face_parser_model_name: Model name for face parser
        face_parser_weight_path: Path to face parser weights
        face_weight: Weight for face areas
        background_weight: Weight for background areas
        face_class_indices: Class index(es) that represent face areas
        device: Device for computation
        
    Returns:
        Tuple of (loss_value, hq_face_mask, weight_map)
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the loss function
    criterion = FaceParserWeightedMSELoss(
        face_parser_model_name=face_parser_model_name,
        face_parser_weight_path=face_parser_weight_path,
        face_weight=face_weight,
        background_weight=background_weight,
        face_class_indices=face_class_indices,
        device=device
    )
    
    # Define image preprocessing (ImageNet normalization)
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Resize to consistent size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess images
    try:
        hq_image = Image.open(hq_image_path).convert('RGB')
        lq_image = Image.open(lq_image_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"Error loading images: {e}")
    
    # Transform images to tensors
    hq_tensor = transform(hq_image).unsqueeze(0).to(device)  # [1, 3, H, W]
    lq_tensor = transform(lq_image).unsqueeze(0).to(device)  # [1, 3, H, W]
    
    print(f"HQ image tensor shape: {hq_tensor.shape}")
    print(f"LQ image tensor shape: {lq_tensor.shape}")
    
    # Compute the weighted MSE loss
    with torch.no_grad():
        loss = criterion(hq_tensor, lq_tensor)
        
        # Get face mask from HQ image
        hq_face_mask = criterion._get_face_masks(hq_tensor)  # [1, H, W]
        
        # Create weight map
        weight_map = torch.ones_like(hq_face_mask, dtype=torch.float32) * background_weight
        
        # Handle face class indices
        face_class_indices_list = face_class_indices if isinstance(face_class_indices, list) else [face_class_indices]
        for class_idx in face_class_indices_list:
            face_areas = (hq_face_mask == class_idx)
            weight_map[face_areas] = face_weight
    
    # Remove batch dimension
    hq_face_mask = hq_face_mask.squeeze(0)  # [H, W]
    weight_map = weight_map.squeeze(0)      # [H, W]
    
    # Save face mask and weight map as tensors
    face_mask_path = os.path.join(output_dir, f"{os.path.basename(hq_image_path).split('.')[0]}_face_mask.pt")
    weight_map_path = os.path.join(output_dir, f"{os.path.basename(hq_image_path).split('.')[0]}_weight_map.pt")
    
    torch.save(hq_face_mask, face_mask_path)
    torch.save(weight_map, weight_map_path)
    
    print(f"Weighted MSE Loss: {loss.item():.6f}")
    print(f"Face mask saved to: {face_mask_path}")
    print(f"Weight map saved to: {weight_map_path}")
    
    return loss, hq_face_mask, weight_map
