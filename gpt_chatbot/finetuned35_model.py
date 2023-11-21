import openai
import os

openai.api_key = "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"
response = openai.ChatCompletion.create(
  model="ft:gpt-3.5-turbo-1106:kw::8NDDT028",
  messages = [
        {"role": "system", "content": "You're a real estate expert who explains apartment transactions."},
        {"role": "user", "content": "2023년 광진구 거래량 방향성은 어때?"}
    ]
)
# print(response)
print(response.choices[0].message.content)
