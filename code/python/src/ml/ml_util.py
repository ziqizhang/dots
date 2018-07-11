from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.multiclass import unique_labels
import numpy as np
import os
import pandas


def outputFalsePredictions(pred, truth, index, model_descriptor, dataset_name, outfolder):
    subfolder = outfolder + "/errors"
    try:
        os.stat(subfolder)
    except:
        os.mkdir(subfolder)
    filename="errors-"+dataset_name+"_"+model_descriptor+".csv"
    filename = os.path.join(subfolder, filename)
    file = open(filename, "w")
    file.write(model_descriptor+"\n")
    for p, t, i in zip(pred, truth, index):
        if p == t:
            line = str(i)+","+str(p) + ",ok\n"
            file.write(line)
        else:
            line = str(i)+","+str(p) + ",wrong\n"
            file.write(line)
    file.close()


def save_scores(nfold_predictions, nfold_truth,
                heldout_predictions, heldout_truth,
                nfold_index, heldout_index,
                model_descriptor,
                dataset_name,
                digits, outfolder,
                instance_tags_train= None,
                instance_tags_test= None,
                accepted_ds_tags: list = None):
    pred = nfold_predictions.tolist()
    if heldout_predictions is not None:
        pred=pred+heldout_predictions.tolist()
    truth = list(nfold_truth)
    if heldout_truth is not None:
        truth=truth+list(heldout_truth)

    index=nfold_index
    if heldout_index is not None:
        index = nfold_index+heldout_index
    outputFalsePredictions(pred, truth, index, model_descriptor,dataset_name, outfolder)
    subfolder = outfolder + "/scores"
    try:
        os.stat(subfolder)
    except:
        os.mkdir(subfolder)
    filename = os.path.join(subfolder, "SCORES_%s.csv" % (dataset_name))
    writer = open(filename, "a+")
    writer.write(model_descriptor+"\n")
    if nfold_predictions is not None:
        writer.write(" N-FOLD AVERAGE :\n")
        write_scores(nfold_predictions, nfold_truth, digits, writer, instance_tags_train, accepted_ds_tags)

    if (heldout_predictions is not None):
        writer.write(" HELDOUT :\n")
        write_scores(heldout_predictions, heldout_truth, digits, writer, instance_tags_test, accepted_ds_tags)

    writer.close()


def write_scores(predictoins, truth: pandas.Series, digits, writer,
                 instance_dst_column=None,
                 accepted_ds_tags=None):
    labels = unique_labels(truth, predictoins)
    if accepted_ds_tags is None:
        target_names = ['%s' % l for l in labels]
        p, r, f1, s = precision_recall_fscore_support(truth, predictoins,
                                                      labels=labels)

        line = prepare_score_string(p, r, f1, s, labels, target_names, digits)
        pa, ra, f1a, sa = precision_recall_fscore_support(truth, predictoins,
                                                          average='micro')
        line += "avg_micro,"
        for v in (pa, ra, f1a):
            line += "{0:0.{1}f}".format(v, digits) + ","
        line += '{0}'.format(np.sum(sa)) + "\n"
        pa, ra, f1a, sa = precision_recall_fscore_support(truth, predictoins,
                                                          average='macro')
        line += "avg_macro,"
        for v in (pa, ra, f1a):
            line += "{0:0.{1}f}".format(v, digits) + ","
        line += '{0}'.format(np.sum(sa)) + "\n\n"
        # average

        writer.write(line)

    if accepted_ds_tags is not None:
        for dstag in accepted_ds_tags:
            #writer.write("\n for data from {} :\n".format(dstag))
            subset_pred = []
            subset_truth = []
            for ds, label in zip(instance_dst_column, predictoins):
                if ds == dstag:
                    if isinstance(label, np.ndarray):
                        subset_pred.append(label[0])
                    else:
                        subset_pred.append(label)
            for ds, label in zip(instance_dst_column, truth):
                if ds == dstag:
                    subset_truth.append(label)
            # print("subset_truth={}, type={}".format(len(subset_truth), type(subset_truth)))
            # print("subset_pred={}, type={}".format(len(subset_pred), type(subset_pred)))
            subset_labels = unique_labels(subset_truth, subset_pred)
            target_names = ['%s' % l for l in labels]
            p, r, f1, s = precision_recall_fscore_support(subset_truth, subset_pred,
                                                          labels=subset_labels)

            line = prepare_score_string(p, r, f1, s, labels, target_names, digits)
            pa, ra, f1a, sa = precision_recall_fscore_support(subset_truth, subset_pred,
                                                              average='micro')
            line += "avg_micro,"
            for v in (pa, ra, f1a):
                line += "{0:0.{1}f}".format(v, digits) + ","
            line += '{0}'.format(np.sum(sa)) + "\n"
            pa, ra, f1a, sa = precision_recall_fscore_support(subset_truth, subset_pred,
                                                              average='macro')
            line += "avg_macro,"
            for v in (pa, ra, f1a):
                line += "{0:0.{1}f}".format(v, digits) + ","
            line += '{0}'.format(np.sum(sa)) + "\n\n"

            writer.write(line)

def prepare_score_string(p, r, f1, s, labels, target_names, digits):
    string = ",precision,recall,f1,support\n"
    for i, label in enumerate(labels):
        string = string + target_names[i] + ","
        for v in (p[i], r[i], f1[i]):
            string = string + "{0:0.{1}f}".format(v, digits) + ","
        string = string + "{0}".format(s[i]) + "\n"
        # values += ["{0}".format(s[i])]
        # report += fmt % tuple(values)

    return string

def feature_scale(option, M):
    if option == -1:
        return M

    print("feature scaling, first perform sanity check...")
    if not isinstance(M, np.ndarray) and M.isnull().values.any():
        print("input matrix has NaN values, replace with 0")
        M.fillna(0)

    if option == 0:  # mean std
        M = feature_scaling_mean_std(M)
        if np.isnan(M).any():
            print("scaled matrix has NaN values, replace with 0")
        M = np.nan_to_num(M)
    elif option == 1:
        M = feature_scaling_min_max(M)
        if np.isnan(M).any():
            print("scaled matrix has NaN values, replace with 0")
        M = np.nan_to_num(M)
    else:
        pass

    print("feature scaling done")
    return M

def feature_scaling_mean_std(feature_set):
    scaler = StandardScaler(with_mean=True, with_std=True)
    return scaler.fit_transform(feature_set)

def feature_scaling_min_max(feature_set):
    """
    Input X must be non-negative for multinomial Naive Bayes model
    :param feature_set:
    :return:
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(feature_set)
