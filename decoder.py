import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline)

def init_decoder(decoder_path:str, context_length:int, generation_config:dict={}):
	"""Initializes the decoder
	Args:
		decoder_path (str): the path of decoder
		context_length (int): the total length of tokens (input + output)
		generation_config (dict):. generation_config has the same meaning as huggingface GenerationConfig. Default to {}
						see more: https://huggingface.co/docs/transformers/v4.33.2/en/main_classes/text_generation#transformers.GenerationConfig 
	Returns:
		object, transformers created decoder
	"""


	factor = context_length//4096
 
	model = AutoModelForCausalLM.from_pretrained(
		decoder_path, 
		device_map = "auto", 
		rope_scaling={"type":"dynamic", "factor":factor} if factor > 1 else None,
		use_flash_attention_2 = True,
		torch_dtype = torch.bfloat16
	)	
	model_generation_config = GenerationConfig.from_pretrained(decoder_path)

	# Create a pipeline for text generation
	tokenizer = AutoTokenizer.from_pretrained(decoder_path, use_fast=True)
	decoder = pipeline(
		"text-generation",	model=model,
		tokenizer=tokenizer, 
		generation_config=model_generation_config,
		# temperature = 0.1,
		# **generation_config
	)

	return decoder