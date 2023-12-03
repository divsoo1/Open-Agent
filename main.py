from utils import get_wiki_data, load_args, write_jsonl_file, init_seed
from prompts import *

from decoder import init_decoder
from encoder import ChunkerandEmbedder

from pathlib import Path
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from termcolor import colored


def query(user_question, LLM, chunker_and_embedder, client, show_sources=False):
    query = Keyword_prompt.format(query=user_question)
    response = LLM(query)[0]['generated_text']
    keywords = response.split("Response\n\nKeywords:")[1]

    wiki_data = get_wiki_data(keywords, limit=10)
    for i, d in enumerate(wiki_data):
        d['id'] = i
        d['metadata'] = {k: v for k, v in d.items() if k != 'summary'}
        d['metadata']['source'] = 'wikipedia'
        d['metadata']['title'] = d['title']
        d['metadata']['url'] = d['url']
        d['metadata']['content'] = d['content']
        del d['title']
        del d['url']
        del d['content']

    vectorized_chunks = chunker_and_embedder.chunk_and_embed(wiki_data, 'summary', chunk_size=512, client=client)
    query_embedding = np.array(client.encode([user_question])[0])

    for d in vectorized_chunks:
        document_embedding = np.array(d['metadata']['embeddings']).reshape(1, -1)
        d['metadata']['similarity'] = cosine_similarity(query_embedding.reshape(1, -1), document_embedding)[0][0]

    sorted_data = sorted(vectorized_chunks, key=lambda x: x['metadata']['similarity'], reverse=True)

    chunks = [d['key'] for d in sorted_data[:4]]
    context = '\n'.join(chunks)

    final_prompt = QA_PROMPT_TEMPLATE.format(context=context, query=user_question)
    final_answer = LLM(final_prompt)[0]['generated_text']
    final_answer = final_answer.split("Helpful Answer:\n\t\t")[1]

    if show_sources:
        sources = chunks
        return final_answer, sources
    else:
        return final_answer


if __name__ == '__main__':
    recipe_path = Path(__file__).parent.resolve()/'config'/'ixl.llama2_13b.key.yaml'
    args = load_args(recipe_path)
    init_seed(args)
    decoder_path = args.model.decoder.path
    context_length = args.model.decoder.context_length
    ENCODER_PATH = args.model.encoder.path

    LLM = init_decoder(decoder_path, context_length)
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_PATH)
    client = SentenceTransformer(ENCODER_PATH)
    chunker_and_embedder = ChunkerandEmbedder(tokenizer)
    show_sources = False
    
    while True:
            user_question = input("Ask a question (type 'exit' to quit): ")

            if user_question.lower() == 'exit':
                break

            final_answer = query(user_question, LLM, chunker_and_embedder, client, show_sources)

            if show_sources:
                print(colored("Answer: " + final_answer[0], 'green'))
                print(colored("Sources: " + final_answer[1], 'blue'))
            else:
                print(colored("Answer: " + final_answer, 'green'))
            # query_prompt = QA_DECODER_PROMPT_TEMPLATE.format(query=user_question)
            # prediction = LLM(user_question)[0]['generated_text']
            # print(prediction)
            # # prediction = prediction.split("Helpful Answer:\n\t\t")[1]
 