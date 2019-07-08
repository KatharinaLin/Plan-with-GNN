def error_calculation(outputs, state_tests):
    errors_precision = []
    errors_recall = []
    errors_acc = []
    f_score = []
    state_tests = state_tests[1:]
    index = 0
    prev = 0
    for output,state in zip(outputs,state_tests):
        if index == 0:
            prev = state
        else:
            if (prev==state).all():
                break
        index += 1
        prev = state
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for out, st in zip(output,state):
            temp = 1
            if st != 1:
                temp = 0
            if temp == 1 and np.round(out[0]) == 1:
                tp += 1
            if temp == 0 and np.round(out[0]) == 0:
                tn += 1
            if temp == 0 and np.round(out[0]) != 0:
                fp += 1
            if temp == 1 and np.round(out[0]) != 1:
                fn += 1
        pre = tp/(fp+tp)
        recall = tp/(tp+fn)
        errors_precision.append(pre)
        errors_recall.append(recall)
        errors_acc.append((tp+tn)/(tp+tn+fp+fn))
        f_score.append((2*pre*recall)/(pre+recall))
    return sum(errors_precision)/index,sum(errors_recall)/index,sum(errors_acc)/index,sum(f_score)/index