---
library_name: peft
---
## Training procedure


The following `bitsandbytes` quantization config was used during training:
- quant_method: gptq
- bits: 4
- tokenizer: None
- dataset: None
- group_size: 128
- damp_percent: 0.1
- desc_act: False
- sym: True
- true_sequential: True
- use_cuda_fp16: False
- model_seqlen: None
- block_name_to_quantize: None
- module_name_preceding_first_block: None
- batch_size: 1
- pad_token_id: None
- disable_exllama: True
- max_input_length: None
### Framework versions


- PEFT 0.5.0
