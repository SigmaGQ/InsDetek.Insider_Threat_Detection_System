import pandas as pd
import numpy as np
from sklearn.metrics import *
import torch
class Evaluation():
    def __init__(self, model, input_data: torch.utils.data.dataloader.DataLoader, lossfunction):
        """Return a list of clean(0)/insider(1) label, for each activities within the seq
        Args:
            model (nn.Module): model for prediction
            input_data (DataLoader): input_data, both feature and label, in dataloader
        """
        self.model = model
        self.dataset = input_data
        self.loss_function = lossfunction
        self.para = False
    
    def __call__(self,  mode):
        """ set the value of logits: [log(prob_i)], predictions: y^, labels: y, avg_loss
        Args:
            mode (String): indicates which dataset to use, train, valid or test
        """
        if mode == 'train':
            self.data = self.dataset.train
        elif mode == 'valid':
            self.data = self.dataset.valid
        elif mode == 'test':
            self.data = self.dataset.test
        else:
            raise ValueError("mode should be 'train', 'valid' or 'test'") 

        with torch.no_grad(): # turns off automatic differentiation, which isn't required but helps save memory
            self.model.eval()

            self.log_prob, self.predictions, self.labels = [], [], []
            total_loss = 0
            for feature_seqs, label_seqs, mask_seqs in self.data:
                seq_len = feature_seqs.shape[1]
                mask_seqs = mask_seqs.bool()
                output_seqs = self.model(feature_seqs) # output_seqs.shape = [batchsize, seq_len, num_class]

                batch_loss_seqs = self.loss_function(output_seqs.reshape([-1,self.dataset.num_class,seq_len]), label_seqs) # loss.shape = [batchsize, seq_len] = [20,72]
                total_loss += torch.mul(batch_loss_seqs, mask_seqs).reshape(-1).sum() # add sum of loss within one batch 
                batch_loss = 0

                real_label_seqs = label_seqs[mask_seqs]
                real_output_seqs = output_seqs[mask_seqs] # real_output_seqs = [len(all real data within the batch)), num_class]
                pred_seqs = pd.DataFrame(real_output_seqs.tolist()).idxmax(axis=1) # pred_seqs = [len(all real data)]
                self.log_prob += real_output_seqs.tolist()
                self.y_prob = np.exp(np.array(self.log_prob)[:,1])
                self.predictions += pred_seqs.tolist()
                self.labels += real_label_seqs.tolist()

            self.model.train()

            self.avg_loss = total_loss / len(self.labels)
            self.para = True

            return self


    def get_metrics(self, print_report = False):
        """
        Get recall: how many insider threats of all insider threats are detected (high → won't miss any insider threats)
            fpr: how many clean activities of all clean activities are misjudged (low → clean users won't be annoyed)
        """
        if not self.para: # must run the __call__ to get the parameters
            raise AttributeError("Must call the evaluation function first (__call__ of Evaluation object)") 

        self.precision, self.recall, self.fscore, self.support = precision_recall_fscore_support(self.labels, self.predictions, beta=2, zero_division = 0)
        tn, fp, fn, tp = confusion_matrix(self.labels, self.predictions).ravel()
        self.tpr = tp / (tp + fn)
        self.fpr = fp / (tn + fp)
        self.acc = (tp + tn) / (tn + fp + fn + tp)
        if print_report:
            print(classification_report(self.labels, self.predictions, zero_division=0))

    # Compute ROC curve and ROC area for each class
    def roc_curve(self):
        if not self.para: # must run the __call__ to get the parameters
            raise AttributeError("Must call the evaluation function first (__call__ of Evaluation object)") 

        if self.dataset.num_class != 2:
            print('ROC curve for multi-label not available')
            return
        else:
            self.y_prob = np.exp(np.array(self.log_prob)[:,1])
            fpr, tpr, threshold = roc_curve(self.labels, self.y_prob)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(8, 8))
            plt.plot(fpr, 
                    tpr, 
                    color = 'darkorange',
                    lw = 2, 
                    label = 'ROC curve (area = %0.3f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
            plt.xlim([0.0, 1.00])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc = "lower right")
            plt.show()

    def pr_curve(self):
        if not self.para: # must run the __call__ to get the parameters
            raise AttributeError("Must call the evaluation function first (__call__ of Evaluation object)") 

        #calculate precision and recall
        precision, recall, thresholds = precision_recall_curve(self.labels, self.y_prob)

        #create precision recall curve
        plt.figure(figsize=(10, 10))
        plt.plot(recall, precision, color='purple')

        #add axis labels to plot
        plt.title('Precision-Recall Curve')
        plt.ylabel('Precision')
        plt.xlabel('Recall')

        #display plot
        plt.show()