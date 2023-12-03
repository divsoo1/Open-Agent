import sentence_transformers

class ChunkerandEmbedder:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def extract_passages_from_tokenized_string(self, s, n, newline_token=None):
        DELIMITERS = ['.', '?', '!']
        if not newline_token is None: DELIMITERS.append(newline_token)

        if len(s) <= n:
            return [s]

        locs = []
        for loc, val in zip(range(n), s):
            if val in DELIMITERS: locs.append(loc)
        idx = max(locs) if locs else n-1

        return [s[0:idx + 1], *self.extract_passages_from_tokenized_string(s[idx + 1:], n)]

    def chunk_and_embed(self, dataset, key_name, chunk_size=512, client=None):
        processed_data = []
        keys_to_embed = []

        for data in dataset:
            key_tokens = self.tokenizer.tokenize(data[key_name])
            keys = self.extract_passages_from_tokenized_string(key_tokens, chunk_size)
            keys = [self.tokenizer.convert_tokens_to_string(key) for key in keys]
            data_id = data['id']

            for key in keys:
                if client:
                    keys_to_embed.append(key)
                processed_data.append({"key": key, "metadata": {**data["metadata"], "id": data_id}})

        if keys_to_embed and client:
            key_embeddings = self.embed_documents_flan(client, keys_to_embed)
            for i, embedding in enumerate(key_embeddings):
                processed_data[i]["metadata"]["embeddings"] = embedding

        return processed_data

    def embed_documents_flan(self, client, texts, parallel=False):
        embed_instruction = "Represent the document for the purpose of similarity search: "
        instruction_pairs = [embed_instruction + text for text in texts]

        if parallel:
            pool = client.start_multi_process_pool()
            print("Starting multiprocess pooling")
            embeddings = client.encode_multi_process(instruction_pairs, pool)
            sentence_transformers.SentenceTransformer.stop_multi_process_pool(pool)
        else:
            print("Encoding sequentially")
            embeddings = client.encode(instruction_pairs)

        return embeddings.tolist()
