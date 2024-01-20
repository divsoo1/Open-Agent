# Open-Agent

Although Large language models (LLMs) have achieved impressive results on multiple natural language tasks, they suffer from problems such as hallucination, especially for knowledge intensive tasks like question answering. We add a dynamic knowledge retrieval method to a pretrained LLM to improve its performance in question answering, by giving it access to an external API. Our results show, that the access to external knowledge improves the performance for questions regarding events more recent than the LLM was trained on, while the performance on older questions is very similar to the baseline LLM
