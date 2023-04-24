import time

import pandas as pd
import argparse
import os
from tqdm import tqdm
import openai

# from openai.embeddings_utils import get_embedding

gpt_model = {
    'curie': 'text-curie-001',
    'davinci': 'text-davinci-003',
    'ada_embed': "text-embedding-ada-002"
}

from configs.config import arg_config_dev as config
# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = config['openai_api_key']
openai.organization = config['openai_organization']

embedding_model = gpt_model['ada_embed']


def get_multi_embeddings_gpt(title_list):
    """
    :param title_list:
    :return:
    """
    response = openai.Embedding.create(
        model=gpt_model['ada_embed'],
        input=title_list
    )
    # print(response)
    embed_list = [r['embedding'] for r in response['data']]
    return embed_list


def parse_args():
    parser = argparse.ArgumentParser()
    # 配置文件路径
    parser.add_argument('--env', type=str, default='dev', help='running of environment: release/dev.')
    parser.add_argument('--date', type=str, default='20230226', help='running date of datasets')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.env == 'release':
        from configs.config import arg_config_release as params
    else:
        from configs.config import arg_config_dev as params

    # 参数定义
    dt = args.date  # 用该天的数据当验证集
    data_path = os.path.join(params['data_path'], 'xzp_la_album_recable_{dt}.csv'.format(dt=dt))
    result_path = os.path.join(params['result_path'], 'album_gpt_embed_{dt}_part3.csv'.format(dt=dt))

    album_df = pd.read_csv(data_path, '\t', error_bad_lines=False)
    # 预处理
    album_df['albumname'].fillna('', inplace=True)
    album_df['tagname2'] = album_df['tagname2'].apply(lambda x: '有声小说' if x == '有声小说1' else x)
    album_df['albumname_clean'] = album_df['albumname'].apply(
        lambda x: x.replace('@BI_COLUMN_SPLIT@', ' ').replace('@BI_FIELD_R@', ' ').replace('@BI_ROW_SPLIT@',
                                                                                           ' ').replace('\u3000',
                                                                                                        ' '))  # 去除制表符、空白符
    # album_df['words'] = album_df['albumname_clean'] + ' ' + album_df['tagname2'].fillna('')

    # album_df['global_vec'] = album_df['words'].apply(get_embedding_gpt)

    index2albumid = album_df['albumid'].tolist()
    index2album = album_df['albumname_clean'].tolist()

    gpt_model = {
        'curie': 'text-curie-001',
        'davinci': 'text-davinci-003',
        'gpt': 'gpt-3.5-turbo'
    }

    with open('data/openai_answer.txt', "w+", encoding='utf-8') as f:
        for index,albumname in enumerate(index2album):
            # if index <10:
            #     continue
            prompt = "'《{}》'的标签有哪些？至少返回6个，不需要换行，用逗号分开".format(albumname)
            completion = openai.ChatCompletion.create(
                model=gpt_model['gpt'],
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            text_raw = completion.choices[0].message['content']
            text = text_raw.strip()
            f.writelines("{}:{}:{}\n".format(index2albumid[index], albumname, text))
            print("Human: {}".format(prompt))
            print("AI: {}".format(text))
            time.sleep(2)

    # n = 0
    # page_size = 200
    # page_num = int(len(index2albumid) / page_size)
    # print('total page: {}'.format(page_num))
    # # 先用文件处理的方式
    # with open(result_path, 'a+', encoding='utf-8') as f:
    #     for num in range(529, page_num):
    #         t1=time.time()
    #         start = (num - 1) * page_size
    #         end = num * page_size
    #         query_list = index2words[start:end]
    #         # print(query_list)
    #         # 通过gpt接口获得embedding
    #         print("Loading page {} [{},{})".format(num, start, end))
    #         try:
    #             embedding_list = get_multi_embeddings_gpt(query_list)
    #         except Exception as e:
    #             print("Error page {} [{},{}):{} failed，retry loading........".format(num,start, end, albumid_list))
    #             time.sleep(60)
    #             embedding_list = get_multi_embeddings_gpt(query_list)
    #             print("Reloading success ！！！".format(start, end))
    #         embedding_str_list = ['='.join(str(i) for i in x) for x in embedding_list]
    #         albumid_list = index2albumid[start:end]
    #         # 写入专辑对应的embedding
    #         for albumid, embedding in zip(albumid_list, embedding_str_list):
    #             f.writelines('{},{}\n'.format(albumid, embedding))
    #         if params['is_local']:
    #             n+=1
    #             if n % 200 == 0: print('{} page has done'.format(n))
    #         time.sleep(1)
    #         print('run time: {}s'.format(time.time() - t1))
    # print('========================Result file saved success {}=================='.format(result_path))
