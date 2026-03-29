from groq import Groq

client = Groq(api_key="")

chat = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "Hello"}]
)

print(chat.choices[0].message.content)
