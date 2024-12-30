from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch


model_ckpt = '/afs/zengwang/ckpt/Qwen2-VL-7B-Instruct'
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_ckpt,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained(model_ckpt)


video_path = '/afs/zengwang/projects/task_define_service/data/video_event/push-up_2.mp4'
time_spot = [2, 15, 21, 43, 51]
query = 'Please narrate the video in real time.'



messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path,
                "max_pixels": 256 * 28 * 28,
                "fps": 5.0,
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]

import pdb; pdb.set_trace()

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)