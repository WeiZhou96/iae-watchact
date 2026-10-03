"""Minimal multi-GPU Qwen3-VL wrapper with the same message layout as teb.backends.LocalVLM
(FRAME i TIME t text before each image, then the task text; greedy decoding)."""
import time
import torch
from PIL import Image


class VLM:
    def __init__(self, model_dir, max_pixels=1638400, device_map='auto', family='qwen', max_memory=None):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.family = family
        if family == 'qwen':
            self.processor = AutoProcessor.from_pretrained(model_dir, max_pixels=max_pixels, local_files_only=True)
        else:  # InternVL (HF-native): one 448-pixel tile per frame, the model's standard video setting
            self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.bfloat16, device_map=device_map, max_memory=max_memory,
                                                                 local_files_only=True).eval()
        self.model_dir = model_dir

    def generate(self, system, frames, text, max_new_tokens=1024):
        content, images = [], []
        for i, (path, ts) in enumerate(frames):
            content.append({'type': 'text', 'text': f'FRAME {i} TIME {ts:.6f} seconds'}); content.append({'type': 'image'})
            with Image.open(path) as im: images.append(im.convert('RGB').copy())
        content.append({'type': 'text', 'text': text})
        msgs = [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}]
        prompt = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        extra = {} if self.family == 'qwen' else {'crop_to_patches': False}
        inputs = self.processor(text=[prompt], images=images, padding=True, return_tensors='pt', **extra).to(self.model.device)
        t0 = time.time()
        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        out = self.processor.batch_decode(ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        return out, {'input_tokens': int(inputs['attention_mask'].sum()), 'seconds': round(time.time() - t0, 2)}
