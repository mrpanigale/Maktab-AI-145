"""Building a unified client to connect to the GapGPT API"""
#==========imports=========
from openai import OpenAI

#========API-KEY=========
API_KEY = "sk-K291sRr7k6jQPFjr5k3xMTzRRr7b16oHVF9r7u25fQaUgo1B"

#==========create-client-obj=============
client = OpenAI(
    base_url="https://api.gapgpt.app/v1",
    api_key=API_KEY
)

#============return-response============ **kwargs --> for tuning response
def complete_gap(prompt, temperature=0, model="gpt-4o-mini", **kwargs):
    """retrun response"""

    return client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": str(prompt)}],
        **kwargs
    )
