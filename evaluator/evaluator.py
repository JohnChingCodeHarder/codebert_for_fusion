# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import logging
import sys
import json
import numpy as np

def read_answers(filename):
    answers={}
    with open(filename) as f:
        for line in f:
            line=line.strip()
            js=json.loads(line)
            answers[js['idx']]=js['target']
    return answers

def read_predictions(filename):
    predictions={}
    with open(filename) as f:
        for line in f:
            line=line.strip()
            idx,label=line.split()
            predictions[int(idx)]=int(label)
    return predictions

# def calculate_scores(answers,predictions):
#     Acc=[]
#     for key in answers:
#         if key not in predictions:
#             logging.error("Missing prediction for index {}.".format(key))
#             sys.exit()
#         Acc.append(answers[key]==predictions[key])

#     scores={}
#     scores['Acc']=np.mean(Acc)
#     return scores

def calculate_scores(answers, predictions):
    # 初始化混淆矩阵所需的变量
    TP = 0  # True Positive
    FP = 0  # False Positive
    TN = 0  # True Negative
    FN = 0  # False Negative
    
    for key in answers:
        if key not in predictions:
            logging.error("Missing prediction for index {}.".format(key))
            sys.exit()
            
        true_label = answers[key]
        pred_label = predictions[key]
        
        if true_label == 1 and pred_label == 1:
            TP += 1
        elif true_label == 0 and pred_label == 1:
            FP += 1
        elif true_label == 0 and pred_label == 0:
            TN += 1
        elif true_label == 1 and pred_label == 0:
            FN += 1
    
    # 计算各项指标
    scores = {}
    
    # 准确率 Accuracy
    scores['Accuracy'] = (TP + TN) / (TP + TN + FP + FN)
    
    # 精确率 Precision
    scores['Precision'] = TP / (TP + FP) if (TP + FP) > 0 else 0
    
    # 召回率 Recall
    scores['Recall'] = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # F1 分数
    if scores['Precision'] + scores['Recall'] > 0:
        scores['F1'] = 2 * scores['Precision'] * scores['Recall'] / (scores['Precision'] + scores['Recall'])
    else:
        scores['F1'] = 0
        
    # 特异度 Specificity
    scores['Specificity'] = TN / (TN + FP) if (TN + FP) > 0 else 0
    
    # Matthews相关系数 MCC
    numerator = (TP * TN) - (FP * FN)
    denominator = ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
    scores['MCC'] = numerator / denominator if denominator != 0 else 0

    return scores

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for Defect Detection dataset.')
    parser.add_argument('--answers', '-a',help="filename of the labels, in jsonl format(original splited dataset jsonl file).")
    parser.add_argument('--predictions', '-p',help="filename of the leaderboard predictions, in txt format(predictions.txt file of different dataset).")
    

    args = parser.parse_args()
    answers=read_answers(args.answers)
    predictions=read_predictions(args.predictions)
    scores=calculate_scores(answers,predictions)
    for (key, value) in scores.items():
        print(f"{key}: {value:.4f}")
    # print(scores)

if __name__ == '__main__':
    main()
