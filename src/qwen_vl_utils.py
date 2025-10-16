"""
Simple wrapper for qwen_vl_utils to ensure compatibility.
If qwen_vl_utils is not installed, this provides a fallback.
"""

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("Warning: qwen_vl_utils not found. Using fallback implementation.")
    
    def process_vision_info(messages):
        """
        Fallback implementation of process_vision_info.
        This is a simplified version for compatibility.
        """
        # Extract images from messages
        images = []
        processed_messages = []
        
        for message in messages:
            if message["role"] == "user" and isinstance(message["content"], list):
                content = message["content"]
                for item in content:
                    if item["type"] == "image" and "image" in item:
                        images.append(item["image"])
                        # Replace with placeholder
                        item["image"] = "<image>"
            processed_messages.append(message)
        
        return processed_messages, images
