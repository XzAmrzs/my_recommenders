import os
import openai

from configs.config import arg_config_dev as config

openai.api_key = config['openai_api_key']
openai.organization = config['openai_organization']

start_sequence = "\nAI:"
restart_sequence = "\nHuman: "

# albumname = '听抖音歌曲最火的歌2023dj'
prompt="'《{}》'的标签有哪些？按照'作品名:标签1,标签2'的格式至少返回6个标签".format('斗破苍穹')
# prompt = "最近5年，深度学习推荐算法的经典论文有哪些？"

gpt_model = {
    'curie': 'text-curie-001',
    'davinci': 'text-davinci-003',
    'gpt': 'gpt-3.5-turbo'
}
#
# response = openai.Completion.create(
#   model=gpt_model['gpt'],
#   prompt=prompt,
#   temperature=0.9,
#   max_tokens=150,
#   top_p=1,
#   frequency_penalty=0,
#   presence_penalty=0.6,
#   stop=[" Human:", " AI:"]
# )

# models = openai.Model.list()

# print the first model's id
# print(models.data[0].id)

completion = openai.ChatCompletion.create(
    model=gpt_model['gpt'],
    messages=[
        {"role": "user", "content": prompt}
    ]
)
#
text_raw = completion.choices[0].message['content']
text = text_raw.strip()

print("Human: {}".format(prompt))
print("AI: {}".format(text))

# with open('../../Rank/test/openai_answer.txt', "w+", encoding='utf-8') as f:
#     f.writelines(text)
