Keyword_prompt = """
### System Prompt
You task is to give keywords based on a query. The keywords should be something like the following example
Example: 
Query: What was the president of South Africa in 2012? 
Keywords: South Africa, president, 2012

Now, for the following query, give keywords:

### User Input
Query: {query}


### Response

"""


QA_PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end. Just provide a single word or a brief one-line response without any explanations.

		{context}

		Question: {query}
		Helpful Answer:
"""

QA_DECODER_PROMPT_TEMPLATE = """Answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.


		{query}
		Helpful Answer:
"""