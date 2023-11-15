import pickle
import os

def save_args_file(args):
    with open(args['model_folder'] + '/config', 'wb') as handle:
        pickle.dump(args, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_args_file(model_folder):
    with open(model_folder + '/config', 'rb') as f:
        args = pickle.load(f)
    return args

def save_model(args):
    command_line = 'cp ./' + args['model_type'] + '/model.py ' + args['model_folder'] + '/' 
    os.system(command_line)

def metrices(preds, labels, num_classes):
    print(preds)
    acc = []
    precision = []
    recall = []
    f1 = []
    count = [] 
    
    TP_all = 0
    FP_all = 0
    TN_all = 0
    FN_all = 0

    for target in range(1, num_classes):
        cnt = 0
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(len(preds)):
            if labels[i] == target:
                cnt += 1
            if preds[i] == target and labels[i] == target:
                TP += 1
                TP_all += 1
            elif preds[i] == target and labels[i] != target:
                FP += 1
                FP_all += 1
            elif preds[i] != target and labels[i] != target:
                TN += 1
                TN_all += 1
            elif preds[i] != target and labels[i] == target:
                FN += 1
                FN_all += 1
        acc.append((TP+TN)/(TP+FP+TN+FN+1e-10))
        precision.append((TP)/(TP+FP+1e-10))
        recall.append(TP/(TP+FN+1e-10))
        f1.append(2 * (precision[target-1] * recall[target-1])/(precision[target-1] + recall[target-1] + 1e-10))
        count.append(cnt)
    
    num_exist_class = len([1 for c in count if c!=0]) + 1e-10
    return sum(acc)/len(count), sum(precision)/num_exist_class, sum(recall)/num_exist_class, sum(f1)/num_exist_class