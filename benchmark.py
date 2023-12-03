from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np

from ..decoder import init_decoder
from ..utils import load_args, write_jsonl_file, init_seed
from ..encoder import ChunkerandEmbedder
from evaluator import ReferencedEvaluator
from ..prompts import *
from ..main import query

def calculate_avg_metrics(metrics_list):
    if not metrics_list:
        return {}

    avg_metrics = {
        "bleu1": np.mean([m["bleu1"] for m in metrics_list]),
        "bleu2": np.mean([m["bleu2"] for m in metrics_list]),
        "bleu3": np.mean([m["bleu3"] for m in metrics_list]),
        "rouge1": np.mean([m["rouge1"] for m in metrics_list]),
        "rouge2": np.mean([m["rouge2"] for m in metrics_list]),
        "rougeL": np.mean([m["rougeL"] for m in metrics_list]),
        "rougeLsum": np.mean([m["rougeLsum"] for m in metrics_list]),
        "precision": np.mean([m["precision"] for m in metrics_list]),
        "recall": np.mean([m["recall"] for m in metrics_list]),
        "f1": np.mean([m["f1"] for m in metrics_list]),
    }

    return avg_metrics

def process_results(data, model, evaluator, tokenizer, chunker_and_embedder, client, jsonl_path):
    results_list = []
    for index, row in data.iterrows():

        if pd.notnull(row['text']):
            if model == "llm":
                query_prompt = QA_DECODER_PROMPT_TEMPLATE.format(query=row['question'])
                prediction = LLM(query_prompt)[0]['generated_text']
                prediction = prediction.split("Helpful Answer:\n\t\t")[1]
                
            elif model == "agent":
                prediction = query(row['question'], LLM, chunker_and_embedder, client, show_sources=False)

            reference = row['text']
            metrics = evaluator.evaluate_all_metrics(prediction, reference)

            result = {
                "id": index,
                "question": row['question'],
                "reference_answer": reference,
                "predicted_answer": prediction,
                "metrics": metrics,
            }
            results_list.append(result)

            avg_metrics = calculate_avg_metrics([m["metrics"] for m in results_list])
            result["avg_metrics"] = avg_metrics

            write_jsonl_file(jsonl_path, [result], mode = "a")


if __name__ == "__main__":

    recipe_path = Path(__file__).parent.parent.resolve()/'config'/'ixl.llama2_13b.key.yaml'
    args = load_args(recipe_path)
    init_seed(args)
    decoder_path = args.model.decoder.path
    context_length = args.model.decoder.context_length
    ENCODER_PATH = args.model.encoder.path

    global LLM
    LLM = init_decoder(decoder_path, context_length)

    global tokenizer, client
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_PATH)
    client = SentenceTransformer(ENCODER_PATH)

    global chunker_and_embedder, evaluator
    chunker_and_embedder = ChunkerandEmbedder(tokenizer)
    evaluator = ReferencedEvaluator(tokenizer)

    squad_path = 'benchmarks/squad/train-v2.0.csv'
    squad = pd.read_csv(squad_path)
    squad = squad.sample(60, random_state=args.common.seed)

    llm_results_path = 'benchmarks/squad/llm_results.jsonl'
    agent_results_path = 'benchmarks/squad/agent_results.jsonl'

    process_results(squad, model="llm", evaluator=evaluator, tokenizer=tokenizer, chunker_and_embedder=chunker_and_embedder, client=client, jsonl_path=llm_results_path)
    process_results(squad, model="agent", evaluator=evaluator, tokenizer=tokenizer, chunker_and_embedder=chunker_and_embedder, client=client, jsonl_path=agent_results_path)
    
    

