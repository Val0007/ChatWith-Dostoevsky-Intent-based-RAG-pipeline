"""Stage 0 sanity check: prove we can call the OpenAI API."""
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()          # reads OPENAI_API_KEY from .env
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello."}],
)
print(resp.choices[0].message.content)
