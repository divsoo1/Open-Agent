from typing import List, Dict
import wikipedia
import yaml
from types import SimpleNamespace
import json
import os
from transformers import enable_full_determinism

class NestedNamespace(SimpleNamespace):
	def __init__(self, dictionary, **kwargs):
		super().__init__(**kwargs)
		for key, value in dictionary.items():
			if isinstance(value, dict):
				self.__setattr__(key, NestedNamespace(value))
			else:
				self.__setattr__(key, value)


def get_wiki_data(query: str, limit = 5) -> List[Dict]:
    wiki_data = []
    for search_result in wikipedia.search(query, results=limit):
        try:
            page = wikipedia.page(search_result)
        except:
            continue
        wiki_data.append({'title': search_result, 'content': page.content, 'summary': page.summary,
                          'url': page.url})
    return wiki_data


def load_args(recipe_path):
    with open(recipe_path) as f:
        args = yaml.load(f, Loader=yaml.SafeLoader)	
    return NestedNamespace(args)
        
        
def write_jsonl_file(file_path, data, mode='a', encoding=None):
    try:
        with open(file_path, mode, encoding=encoding) as file:
            for entry in data:
                file.write(json.dumps(entry, ensure_ascii=False) + '\n')
    except FileNotFoundError as e:
        raise
        


def init_seed(args):
	os.environ['PYTHONHASHSEED'] = str(args.common.seed)
	enable_full_determinism(args.common.seed, warn_only=True)