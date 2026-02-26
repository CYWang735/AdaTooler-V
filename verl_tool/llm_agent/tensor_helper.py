import torch
from typing import Dict, Tuple, List
from dataclasses import dataclass

@dataclass
class TensorConfig:
    pad_token_id: int
    max_prompt_length: int
    max_obs_length: int
    max_start_length: int
    max_response_length: int

class TensorHelper:
    def __init__(self, config: TensorConfig):
        self.config = config

    def cut_to_effective_len(self, tensor_dict: Dict[str, torch.Tensor], 
                            keys: List[str], cut_left: bool = True) -> Dict[str, torch.Tensor]:
        """Cut tensors to their effective length based on attention mask."""
        effective_len = tensor_dict['attention_mask'].sum(dim=1).max()
        result = tensor_dict.copy()
        
        for key in keys:
            if cut_left:
                result[key] = tensor_dict[key][:, -effective_len:]
            else:
                result[key] = tensor_dict[key][:, :effective_len]
        return result

    def convert_pad_structure(self, tensor: torch.Tensor, pad_to_left: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert padding structure and return sorted tensor with indices."""
        mask = tensor != self.config.pad_token_id if pad_to_left else tensor == self.config.pad_token_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        return tensor.gather(1, sorted_indices), sorted_indices

    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create attention mask from input ids."""
        return torch.where(input_ids != self.config.pad_token_id, 1, 0)

    def create_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Create position ids from attention mask."""
        return (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

    def concatenate_with_padding(self, tensors: List[torch.Tensor], 
                               pad_to_left: bool = True) -> torch.Tensor:
        """Concatenate tensors and handle padding."""
        device = tensors[0].device
        tensors = [tensor.to(device) for tensor in tensors]
        concatenated = torch.cat(tensors, dim=1)
        padded_tensor, _ = self.convert_pad_structure(concatenated, pad_to_left)
        return padded_tensor

    def _example_level_pad(self, responses: torch.Tensor, 
                          responses_str: List[str], 
                          active_mask: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        """
        Pad responses for non-active examples with pad tokens.
        """
        assert active_mask.sum() == responses.shape[0], f"Active mask sum: {active_mask.sum()}, responses shape: {responses.shape}"
        # Create masked responses tensor
        batch_size = active_mask.shape[0]
        
        seq_len = responses.shape[1]
        
        padded_responses = torch.full(
            (batch_size, seq_len), self.config.pad_token_id,
            dtype=responses.dtype, device=responses.device
        )
        padded_responses[active_mask] = responses
        
        # Create masked response strings
        padded_responses_str = [""] * batch_size
        
        s = 0
        for i, is_active in enumerate(active_mask):
            if is_active:
                padded_responses_str[i] = responses_str[s]
                s += 1
                
        return padded_responses, padded_responses_str
    
    def pad_tensor(self, tensor: torch.Tensor, max_length: int, padding_side: str = "right") -> torch.Tensor:
        """
            Pad tensor with pad token id to a specified length in the sequence dimension.
            Args:
                tensor (torch.Tensor): The tensor to pad (batch_size, seq_len).
                max_length (int): The length to pad to.
                padding_side (str): 'right' or 'left' padding.
            Returns:
                torch.Tensor: The padded tensor.    
        """
        pad_token_id = self.config.pad_token_id
        batch_size, seq_len = tensor.shape
        
        if padding_side == "right":
            padded_tensor = torch.full((batch_size, max_length), pad_token_id, dtype=tensor.dtype, device=tensor.device)
            padded_tensor[:, :seq_len] = tensor
        elif padding_side == "left":
            padded_tensor = torch.full((batch_size, max_length), pad_token_id, dtype=tensor.dtype, device=tensor.device)
            padded_tensor[:, -seq_len:] = tensor
        else:
            raise ValueError("padding_side must be either 'right' or 'left'")
        
        return padded_tensor