from sklearn.metrics import *
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import numpy as np


class ModelMetrics:
    def __init__(self,
                 y_true: list,
                 y_pred_scores: list,
                 threshold=0.5,
                 y_pred=None,
                 model=None,
                 save_dir=None):

        self.y_true = np.array(y_true)
        self.y_pred_scores = np.array(y_pred_scores)
        self.threshold = threshold
        self.model = model
        self.save_dir = save_dir if save_dir is not None else "./"

        if y_pred is None:
            self.y_pred = self.get_y_pred()
        else:
            self.y_pred = y_pred

        self.lw = 1.3

    def get_roc_curve(self,
                      figsize: tuple = (8, 8),
                      show: bool = False):
        """
        :return: auc score of receiver operating characteristic curve
        """

        fpr, tpr, threshold = roc_curve(self.y_true, self.y_pred_scores)

        plt.figure(figsize=figsize)
        plt.plot(fpr,
                 tpr,
                 'k-',
                 lw=self.lw,
                 label='ROC AUC Score (area = %0.2f)' % auc(fpr, tpr))

        plt.plot([0, 1], [0, 1], color='black', linewidth=self.lw, linestyle='-.')
        plt.xlim([-0.005, 1.0])
        plt.ylim([0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=13)
        plt.ylabel('True Positive Rate', fontsize=13)
        plt.title('ROC curve')
        plt.legend(loc="lower right")

        if self.save_dir is not None:
            plt.savefig(os.path.join(self.save_dir, "roc_curve.jpg"), bbox_inches='tight')

        if show:
            plt.show()
            plt.close()

        return dict(
            fpr=fpr,
            tpr=tpr,
            thresholds=threshold,
            auc=auc(fpr, tpr)
        )

    def get_pr_curve(self,
                     figsize: tuple = (8, 8),
                     show: bool = False):
        """
        :return: auc score of precision-recall curve
        """
        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_pred_scores)
        plt.figure(figsize=figsize)
        plt.plot(recall,
                 precision,
                 'k-',
                 lw=self.lw,
                 label='PR AUC Score (area = %0.2f)' % auc(recall, precision))

        plt.xlim([-0.005, 1.0])
        plt.ylim([0, 1.05])
        plt.xlabel('Recall', fontsize=13)
        plt.ylabel('Precision', fontsize=13)
        plt.title('Precision-Recall curve')
        plt.legend(loc="lower right")

        if self.save_dir is not None:
            plt.savefig(os.path.join(self.save_dir, "pr_curve.jpg"), bbox_inches='tight')

        if show:
            plt.show()
            plt.close()

        return dict(
            precision=precision,
            recall=recall,
            thresholds=thresholds,
            auc=auc(recall, precision)
        )

    def get_y_pred(self):
        if type(self.y_pred_scores) != np.ndarray:
            y_pred_scores = np.array(self.y_pred_scores)
            y_pred = np.where(y_pred_scores > self.threshold, 1, 0)
        else:
            y_pred = np.where(self.y_pred_scores > self.threshold, 1, 0)
        return y_pred

    def get_confusion_matrix(self,
                             show: bool = True,
                             positive_class_name: str = "TCFA > 200",
                             negative_class_name: str = "Negative"):
        conf_matrix = confusion_matrix(self.y_true, self.y_pred)

        if show:
            df_cm = pd.DataFrame(conf_matrix,
                                 index=[i for i in [negative_class_name, positive_class_name]],
                                 columns=[i for i in [negative_class_name, positive_class_name]])
            plt.figure(figsize=(10, 8))
            sns.heatmap(df_cm, annot=True)

            if self.save_dir is not None:
                plt.savefig(os.path.join(self.save_dir, "confusion_matrix.jpg"),
                            bbox_inches='tight')

            # plt.show()
            plt.close()

        return conf_matrix

    def get_ppv_npv(self) -> tuple:
        conf_matrix = confusion_matrix(self.y_true, self.y_pred)

        tp = conf_matrix[1, 1]
        tn = conf_matrix[0, 0]
        fp = conf_matrix[0, 1]
        fn = conf_matrix[1, 0]

        ppv = tp / (tp + fp)
        npv = tn / (tn + fn)

        return ppv, npv

    def get_sen_spe(self) -> tuple:
        """
            Get Sensitivity and Specificity
        :return:
            sensitivity, specificity
        """
        conf_matrix = confusion_matrix(self.y_true, self.y_pred)

        tp = conf_matrix[1, 1]
        tn = conf_matrix[0, 0]
        fp = conf_matrix[0, 1]
        fn = conf_matrix[1, 0]

        sen = tp / (tp + fn)
        spe = tn / (tn + fp)

        return sen, spe

    def get_f1_score(self):
        return f1_score(self.y_true, self.y_pred, average='micro', pos_label=1)

    def get_precision_score(self):
        conf_matrix = confusion_matrix(self.y_true, self.y_pred)

        tp = conf_matrix[1, 1]
        fp = conf_matrix[0, 1]

        precision = tp / (tp + fp)
        return precision

    def get_accuracy_score(self):
        return accuracy_score(self.y_true, self.y_pred)

    def get_recall_score(self):
        conf_matrix = confusion_matrix(self.y_true, self.y_pred)

        tp = conf_matrix[1, 1]
        fn = conf_matrix[1, 0]

        recall = tp / (tp + fn)
        return recall
        # return recall_score(self.y_true, self.y_pred, average='micro', pos_label=1)

    def print_all(self):

        _recall_score = self.get_recall_score()
        _precision_score = self.get_precision_score()
        conf_matrix = self.get_confusion_matrix()
        ro_auc_score = self.get_roc_curve()['auc']
        pr_auc_score = self.get_pr_curve()['auc']
        ppv, npv = self.get_ppv_npv()
        sensitivity, specificity = self.get_sen_spe()
        accuracy = self.get_accuracy_score()

        print(f"Precision score : {_precision_score}")
        print(f"Recall score : {_recall_score}")
        print(f"ROC-AUC score : {ro_auc_score}")
        print(f"PR-AUC score : {pr_auc_score}")
        print(f"Accuracy score : {accuracy}")
        print(f"PPV : {ppv}")
        print(f"NPV : {npv}")
        print(f"Sensitivity : {sensitivity}")
        print(f"Specificity : {specificity}")

        result = {
            "recall": _recall_score,
            "precision": _precision_score,
            "roc-auc": ro_auc_score,
            "pr-auc": pr_auc_score,
            "ppv": ppv,
            "npv": npv,
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "confusion_matrix": conf_matrix
        }

        pd.DataFrame.from_dict(result, orient="index").to_csv(os.path.join(self.save_dir, "result.csv"))
        return result


if __name__ == '__main__':
    pass
