# Calling await call_llm(p) one at a time is still sequential.
# We are awaiting each call before starting the next. To run them in parallel we have to use asyncio.gather.

import httpx
import asyncio
from LearningMaterials.config import API_KEY

async def call_llm_async(client: httpx.AsyncClient, prompt: str) -> str:
    response = await client.post("https://api.groq.com/openai/v1/chat/completions",
        json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}]},
        headers={"Authorization": f"Bearer {API_KEY}"})
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

async def main():
    prompts = ["Summarize the plot of The Lord of the Rings.", 
               "What is the capital of France?", 
               "What is the meaning of life?"]
    # gather() fires all coroutines at once and waits for ALL to finish
    # results come back in the SAME ORDER as the input list
    async with httpx.AsyncClient() as client:
        response = await asyncio.gather(*(call_llm_async(client, p) for p in prompts))

    for prompt, answer in zip(prompts, response):
        print(f"Q: {prompt}\nA: {answer}\n")

asyncio.run(main())
