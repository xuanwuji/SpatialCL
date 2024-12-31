#!/usr/bin/env python

# @Time    : 2024/6/1 13:34
# @Author  : Yao Xuan
# @Email   : xuany0512@163.com
# @File    : train_util.py

import random
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib.ticker import MaxNLocator
from lifelines.utils import concordance_index
import scipy
from sklearn.metrics import auc, roc_curve, f1_score, precision_recall_curve, average_precision_score

seed = 20000512


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


setup_seed(seed)


def mask_tokens(tokens, tokenizer, mask_prob=0.15):
    masked_tokens = tokens.clone()
    device = tokens.device

    # 遍历tokens，随机选择一些进行mask

    def creat_mask_matrix():
        b, l = tokens.size()
        # 生成mask_matrix
        mask_matrix = torch.full((b, l), False, device=device)
        # 随机选择一些位置，并将其设置为True
        num_true_elements = int(b * l * mask_prob)
        true_indices = torch.randperm(b * l)[:num_true_elements]
        mask_matrix.view(-1)[true_indices] = True
        return mask_matrix

    def create_special_matrix():
        # 根据pad的位置生成一个矩阵，pad的位置为False，不pad的地方为True
        special_matrix = ~torch.isin(tokens, torch.tensor(tokenizer.all_special_ids, device=device))
        return special_matrix

    mask_matrix = creat_mask_matrix() & create_special_matrix()
    masked_tokens_label = tokens[mask_matrix]
    new_masked_token_label = masked_tokens_label.clone()

    # 对mask的token进行随机替换
    for i, token in enumerate(masked_tokens_label):
        # 80%的概率替换为[MASK] token
        if random.random() < 0.8:
            new_masked_token_label[i] = tokenizer.mask_token_id
        # 10%的概率替换为随机token
        elif random.random() > 0.9:
            new_masked_token_label[i] = torch.randint(4, 29, (1,), device=device).item()
        # 10%的概率保持不变
        else:
            new_masked_token_label[i] = token

    masked_tokens[mask_matrix] = new_masked_token_label

    return masked_tokens, mask_matrix, masked_tokens_label


def create_seg_tok(tokens, len_seq1, len_seq2, device):
    # 不可避免使用for循环
    batch, max_length = tokens.size()
    seg_toks = torch.zeros(batch, max_length, dtype=torch.int64, device=device)
    for i in range(batch):
        seg_0 = torch.zeros(len_seq1[i] + 2, dtype=torch.int64, device=device)
        seg_1 = torch.ones(len_seq2[i] + 1, dtype=torch.int64, device=device)
        seg_p = torch.full((max_length - len(seg_0) - len(seg_1),), 2, dtype=torch.int64, device=device)
        seg_tok = torch.cat((seg_0, seg_1, seg_p), dim=0)
        seg_toks[i] = seg_tok
    return seg_toks


def plot_loss(max_epochs, data, color, label, save_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(x=range(max_epochs), y=data, color=color, label=label)
    ax.set_xlabel('Epoch', fontsize=16)
    ax.set_ylabel('Loss', fontsize=16)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(save_path + '.jpg', dpi=300)
    plt.savefig(save_path + '.svg', format='svg', dpi=300)
    ax.legend(fontsize=14)
    plt.close()


def plot_loss_by_iter(max_iter, data, color, label, save_path, step):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(x=range(max_iter), y=data, color=color, label=label)
    ax.set_xlabel('Iterations (step:' + str(step) + ')', fontsize=16)
    ax.set_ylabel('Loss', fontsize=16)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(save_path + '.jpg', dpi=300)
    plt.savefig(save_path + '.svg', format='svg', dpi=300)
    ax.legend(fontsize=14)


def plot_pcc(max_epochs, data, color, label, save_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(x=range(max_epochs), y=data, color=color, label=label)
    ax.set_xlabel('Epoch', fontsize=16)
    ax.set_ylabel('Pearson Correlation Coefficient', fontsize=16)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(save_path + '.jpg', dpi=300)
    plt.savefig(save_path + '.svg', format='svg', dpi=300)
    ax.legend(fontsize=14)


# def plot_loss_by_step(data, color, label, save_path):
#     fig, ax = plt.subplots(figsize=(7, 5))
#     sns.lineplot(x=range(len(data)), y=data, color=color, label=label)
#     ax.set_xlabel('Step', fontsize=16)
#     ax.set_ylabel('Loss', fontsize=16)
#     ax.tick_params(axis='x', labelsize=14)
#     ax.tick_params(axis='y', labelsize=14)
#     ax.xaxis.set_major_locator(MaxNLocator(5))
#     ax.yaxis.set_major_locator(MaxNLocator(3))
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     plt.savefig(save_path + '.jpg', dpi=300)
#     plt.savefig(save_path + '.svg', format='svg', dpi=300)
#     ax.legend(fontsize=14)

def plot_performance(labels, preds, title, fig_save_path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.set_position([0.12, 0.12, 0.52, 0.77])
    plt.scatter(x=preds, y=labels, s=30, alpha=0.5, color='k', edgecolors='none')
    plt.plot([0, 1.2], [0, 1.2], color='k', linestyle='--')
    pcc = scipy.stats.pearsonr(labels, preds)
    # pcc = np.corrcoef(df['preds'], df['labels'])
    print(pcc)
    scc = scipy.stats.spearmanr(labels, preds)
    print('SCC:' + str(scc[0]))
    mse = mean_squared_error(labels, preds)
    print('MSE:' + str(mse))
    rmse = mean_squared_error(labels, preds, squared=False)
    print('RMSE:' + str(rmse))
    r2 = r2_score(labels, preds)
    print('R2 Score:' + str(r2))
    c_index = concordance_index(labels, preds)
    print('c_index:' + str(c_index))

    ax.set_xlabel('Predictions', fontsize=18)
    ax.set_ylabel('Labels', fontsize=18)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    plt.xlim([0, 1.2])
    plt.ylim([0, 1.2])
    # plt.text(0, -0.07, 'PCC:' + str(pcc[0][1]))
    plt.text(1.25, 0.6, 'PCC:' + str(round(pcc[0], 4)), fontsize=15)
    plt.text(1.25, 0.5, 'p value:' + str(format(pcc[1], '.4g')), fontsize=15)
    plt.text(1.25, 0.4, 'MSE:' + str(round(mse, 4)), fontsize=15)
    plt.text(1.25, 0.3, 'R2 Score:' + str(round(r2, 4)), fontsize=15)
    plt.text(1.25, 0.2, 'SCC:' + str(round(scc[0], 4)), fontsize=15)
    plt.text(1.25, 0.1, 'RMSE:' + str(round(rmse, 4)), fontsize=15)
    plt.text(1.25, 0, 'c_index:' + str(round(c_index, 4)), fontsize=15)
    # title要改
    plt.title(title, fontsize=16)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    # plt.show()
    # plt.title('HLA-A*02:01 Model Prediction vs True values(bs512) val:HLA-B4402',fontsize=16)
    # 文件名要改
    plt.savefig(fig_save_path + '.jpg', dpi=300)
    plt.savefig(fig_save_path + '.svg', format='svg', dpi=300)
    # plt.savefig('Z:/Work/pMHC_prediction/Test/test_affinity/a0201_netmhcpan_bs512_b4402_Correlation.png',dpi=500)
    plt.close()


def calculate_index(df, thr):
    df['b_preds'] = 0
    df.loc[df['preds'] > thr, 'b_preds'] = 1
    df['scanpy_result'] = ''
    df.loc[(df['labels']) == 1 & (df['b_preds'] == 1), 'scanpy_result'] = 'TP'
    df.loc[(df['labels']) == 1 & (df['b_preds'] == 0), 'scanpy_result'] = 'FN'
    df.loc[(df['labels']) == 0 & (df['b_preds'] == 1), 'scanpy_result'] = 'FP'
    df.loc[(df['labels']) == 0 & (df['b_preds'] == 0), 'scanpy_result'] = 'TN'
    # print(df)
    accuracy = (len(df[df['scanpy_result'] == 'TP']) + len(df[df['scanpy_result'] == 'TN'])) / len(df)
    precision = len(df[df['scanpy_result'] == 'TP']) / (len(df[df['scanpy_result'] == 'TP']) + len(df[df['scanpy_result'] == 'FP']))
    recall = len(df[df['scanpy_result'] == 'TP']) / (len(df[df['scanpy_result'] == 'TP']) + len(df[df['scanpy_result'] == 'FN']))
    return accuracy, precision, recall


def plot_index(data, file_name, save_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(data=data[['accuracy', 'precision', 'recall']], palette='colorblind')
    ax.set_xlabel('Epoch', fontsize=16)
    ax.set_ylabel('Evaluation', fontsize=16)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(save_dir + file_name + '.jpg', dpi=300)
    plt.savefig(save_dir + file_name + '.svg', format='svg', dpi=300)
    ax.legend(fontsize=14)


def plot_index_by_iter(data, file_name, save_dir,step):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(data=data[['accuracy', 'precision', 'recall']], palette='colorblind')
    ax.set_xlabel('Iterations (step:' + str(step) + ')', fontsize=16)
    ax.set_ylabel('Evaluation', fontsize=16)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(save_dir + file_name + '.jpg', dpi=300)
    plt.savefig(save_dir + file_name + '.svg', format='svg', dpi=300)
    ax.legend(fontsize=14)


def discrete(ls, thres):
    dis_ls = []
    for p in ls:
        if p > thres:
            p = 1
            dis_ls.append(p)
        else:
            p = 0
            dis_ls.append(p)
    return dis_ls


def plot_roc(fpr, tpr, roc_auc, f1s, legend, ax, thr):
    ax.plot(fpr, tpr, '--',
            label=legend + '(auc = {0:.3f}; F1_score = {1:.3f})'.format(roc_auc, f1s) + 'thr:' + str(thr), lw=2)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel('False Positive Rate', fontsize=18)
    ax.set_ylabel('True Positive Rate', fontsize=18)
    ax.set_xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc="lower right")


def plot_pr(recall, precision, pr_auc, f1s, legend, ax, thr):
    ax.plot(recall, precision, '--',
            label=legend + '(auc = {0:.3f}; F1_score = {1:.3f})'.format(pr_auc, f1s) + 'thr:' + str(thr), lw=2)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel('Recall', fontsize=18)
    ax.set_ylabel('Precision', fontsize=18)
    ax.set_xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc="lower right")


def plot_auc(labels, preds, thr, title, save_path):
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    fig1.suptitle(title + ' ROC Curve', fontsize=18)
    fig2.suptitle(title + ' PR Curve', fontsize=18)
    fpr, tpr, thersholds = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    precision, recall, thersholds = precision_recall_curve(labels, preds)
    pr_auc = auc(recall, precision)
    thres = thr
    preds_dis = discrete(preds, thres)
    f1s = f1_score(labels, preds_dis)
    print('ROC_AUC:'+str(roc_auc))
    print('PR_AUC:'+str(pr_auc))
    plot_roc(fpr, tpr, roc_auc, f1s, '', ax1, thr)
    plot_pr(recall, precision, pr_auc, f1s, '', ax2, thr)
    fig1.savefig(save_path + 'ROC.png', dpi=300)
    fig1.savefig(save_path + 'ROC.svg', format='svg', dpi=300)
    fig2.savefig(save_path + 'PR.png', dpi=300)
    fig2.savefig(save_path + 'PR.svg', format='svg', dpi=300)

    return roc_auc, pr_auc
