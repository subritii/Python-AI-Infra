# asyncio.gather alone fires ALL requests at once — fine for 3 prompts, causes 429 errors at 100.
# A Semaphore fixes this by capping how many requests are in-flight at the same time.

import asyncio
import httpx

MAX_CONCURRENT = 10  # tune this to your API's requests-per-minute limit

async def call_llm(
    client: httpx.AsyncClient,  
    sem: asyncio.Semaphore,     
    prompt: str,
) -> str:
    async with sem:  # acquire a slot — blocks here if 10 requests are already running
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}]  # prompt sent as a user message
            },
            headers={"Authorization": f"Bearer {API_KEY}"},  # API key sent in every request header
        )
        r.raise_for_status()  # throws an error if the API returned 4xx or 5xx
        return r.json()["choices"][0]["message"]["content"]  # extract the text reply from the response
    # semaphore slot is automatically released here, letting the next waiting request proceed

async def main():
    prompts = [f"prompt {i}" for i in range(100)]  # 100 prompts to send
    sem = asyncio.Semaphore(MAX_CONCURRENT)         # create the semaphore with 10 slots

    async with httpx.AsyncClient(timeout=30.0) as client:  # 30s timeout per request
        results = await asyncio.gather(
            *[call_llm(client, sem, p) for p in prompts],  # registers all 100 coroutines, but each one blocks at the semaphore until a slot is free
            return_exceptions=True,  # ... but don't crash everything if one fails; store the error instead
        )
        # results is a list of 100 items — each is either a string reply or an Exception object

    print(f"Got {len(results)} results")

asyncio.run(main())  # entry point — starts the async event loop