import asyncio
import os
from urllib.parse import urljoin

import openai
from aiohttp import ClientSession


def get_llm_backend():
    return os.environ.get("LLAMBO_LLM_BACKEND", "openai").strip().lower()


def get_request_timeout(default=10):
    if get_llm_backend() == "ollama":
        default = 120
    return float(os.environ.get("LLAMBO_REQUEST_TIMEOUT", default))


def configure_openai_from_env():
    if get_llm_backend() != "openai":
        return
    openai.api_type = os.environ["OPENAI_API_TYPE"]
    openai.api_version = os.environ["OPENAI_API_VERSION"]
    openai.api_base = os.environ["OPENAI_API_BASE"]
    openai.api_key = os.environ["OPENAI_API_KEY"]


def _normalize_ollama_host(host):
    host = host or "http://127.0.0.1:11434"
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return host.rstrip("/") + "/"


async def _ollama_one_chat(session, url, payload, timeout):
    async with session.post(url, json=payload, timeout=timeout) as response:
        response.raise_for_status()
        return await response.json()


async def chat_completion(
    *,
    session,
    engine,
    messages,
    temperature,
    max_tokens,
    top_p,
    n,
    request_timeout=None,
):
    backend = get_llm_backend()
    request_timeout = request_timeout or get_request_timeout()

    if backend == "ollama":
        model = engine or os.environ.get("LLAMBO_MODEL", "qwen3.5:9b")
        host = _normalize_ollama_host(os.environ.get("OLLAMA_HOST"))
        url = urljoin(host, "api/chat")
        base_payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        }
        tasks = [
            _ollama_one_chat(session, url, base_payload, request_timeout)
            for _ in range(n)
        ]
        results = await asyncio.gather(*tasks)
        choices = [
            {"message": {"content": result.get("message", {}).get("content", "")}}
            for result in results
        ]
        prompt_tokens = sum(result.get("prompt_eval_count", 0) for result in results)
        completion_tokens = sum(result.get("eval_count", 0) for result in results)
        return {
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    openai.aiosession.set(session)
    return await openai.ChatCompletion.acreate(
        engine=engine,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        n=n,
        request_timeout=request_timeout,
    )


def estimate_cost(response):
    if get_llm_backend() == "ollama":
        return 0
    usage = response["usage"]
    return 0.0015 * (usage["prompt_tokens"] / 1000) + 0.002 * (usage["completion_tokens"] / 1000)
