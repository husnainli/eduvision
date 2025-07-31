import requests

TOGETHER_API_KEY = "6866d79687419256304972259a77b271650459b8d0b22bae21fd59d4265066d4"

system_prompt = (
    "أنت مساعد ذكي يتحدث العربية فقط، وتفهم الوثائق والمحتوى باللغة العربية.\n"
    "يُمنع منعًا باتًا استخدام أي كلمات إنجليزية أو من لغات أخرى في ردودك.\n"
    "يجب أن تكون كل الإجابات باللغة العربية الفصحى وبشكل رسمي وواضح."
)

def query_llama3(prompt, system_prompt=system_prompt):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "meta-llama/Llama-3-8b-chat-hf",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 1024
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if "choices" in result:
            return result["choices"][0]["message"]["content"]
        elif "output" in result:
            return result["output"]
        else:
            raise ValueError(f"Unexpected API response: {result}")
    except Exception as e:
        return f"❌ Error: {e}"
