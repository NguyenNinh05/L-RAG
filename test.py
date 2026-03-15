from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "unsloth/Qwen3-4B-Instruct-2507-GGUF"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    gguf_file="Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
)

# Load model với offload folder
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    gguf_file="Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    dtype=torch.float16,  # ✅ Dùng dtype thay vì torch_dtype
    device_map="auto",
    offload_folder="./offload",  # 👈 Thêm folder tạm để offload
    low_cpu_mem_usage=True  # 👈 Giảm RAM usage
)

# Test
prompt = "Give me a short introduction to large language model. Do you know qwen?"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
content = tokenizer.decode(output_ids, skip_special_tokens=True)

print("content:", content)