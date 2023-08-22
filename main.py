import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from models.modeling_baichuan import BaichuanForCausalLM

device="cuda:6"
tokenizer = AutoTokenizer.from_pretrained("/mnt/datadisk0/public/plms/baichuan13b", use_fast=False, trust_remote_code=True)
model = BaichuanForCausalLM.from_pretrained("/mnt/datadisk0/public/plms/baichuan13b", device_map={"":"cpu"}, torch_dtype=torch.float16, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained("/mnt/datadisk0/public/plms/baichuan13b")
model.to(device)
x="帮我写一篇1000字春天作文"
y="帮我写一篇1000字秋天作文"
messages = []
messages2 = []

messages.append({"role": "user", "content": x})
messages2.append({"role": "user", "content": y})

# ids=tokenizer(x,return_tensors='pt')
# ids=ids.to(device)
# pred =model.generate(
#     ids.input_ids,
# max_new_tokens=512,
# repetition_penalty=1.1
# )
# response = tokenizer.decode(pred.cpu().tolist()[0])
# if response[-4:] == "</s>": response = response[:-4]
#
# print(response)

response = model.chat(tokenizer, messages,messages2)

import time
st=time.time()
response = model.chat(tokenizer, messages,messages2)
et=time.time()
print(et-st)

print(response)