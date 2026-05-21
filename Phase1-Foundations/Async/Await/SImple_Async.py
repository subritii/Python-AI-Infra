import httpx
import asyncio

# A synchronous function to make an API call
def call_llm_sync(prompt: str)-> str:
    with httpx.Client() as client:
        reponse = client.post("https://api.example.com/llm", json={...})
        return response.json()["choices"][0]["message"]["content"]
     
# An asynchronouse function to make an API call
async def call_llm_async(prompt: str) -> str:
    async with httpx.AsyncClient as client:
        response = await client.post("https://api.example.com/llm", json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}]}, headers={"Authorization": f"Bearer {API_KEY}"})
        return response.json()["choices"][0]["message"]["content"]

async def main():
    result = await call_llm_async("What is the capital of France?")
    print(result)

# Outermost call to create a new event loop and hands it the main() coroutine to run
asyncio.run(main())


