# -*- coding: utf-8 -*-#

# Name:     evaluate
# Author:   Jasper.X
# Date:     2021/1/12
# Description:
from tqdm import tqdm


def evaluate_metrics(user_recall_item_dict, user_item_hist_list, topk=10):
    """
    召回效果指标评估函数
    :param user_recall_item_dict:
    :param user_item_hist_list:
    :param topk:
    :return:
    """
    # 准确率和召回率
    hit_num = 0
    rec_count = 0
    test_count = 0

    for user in tqdm(user_recall_item_dict):
        # 获取该JD在测试集中的候选人历史投递列表
        if user in user_item_hist_list:
            hist_items = user_item_hist_list[user]
            # 获取该JD的推荐候选人列表
            tmp_recall_items = [x[0] for x in user_recall_item_dict[user][:topk]]
            for rec_item in tmp_recall_items:
                if rec_item in hist_items:
                    hit_num += 1

            # 计算所推荐候选物品总数
            rec_count += len(tmp_recall_items)
            # 计算测试集中物品总数
            test_count += len(hist_items)

    # 计算准确率 命中个数/所推荐物品个数
    precision = hit_num / (1.0 * rec_count)
    # 计算召回率 命中个数/测试集中候选物品个数
    recall = hit_num / (1.0 * test_count)
    print(' topk: ', topk, ' : ', 'hit_num: ', hit_num, 'hit_rate: ', recall)
    print('precisioin=%.4f\t recall=%.4f\t' % (precision, recall))