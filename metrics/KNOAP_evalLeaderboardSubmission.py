import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

parser = argparse.ArgumentParser()

def calc_metrics(labels,predictions):
    nrSubj = len(predictions)
    estimatedClass = 2 * np.ones(nrSubj, dtype=int)
    iSROA_probability = np.empty(nrSubj, dtype=float)
    iSROA_probability[:]=np.nan
    invalidFlag = False
    for s in range(nrSubj):
        kneeMask = labels['Knee'].iloc[s] == predictions['Knee']
        currKnee = predictions[kneeMask]

        # if subject is missing
        if currKnee.shape[0] == 0:
            print('WARNING: Knee %s missing from predictions' % labels['Knee'].iloc[s])
            invalidFlag = True
            continue

        currKnee = currKnee.reset_index(drop=True)
        pCTRL = currKnee['Control_probability'].iloc[0]
        pISROA = currKnee['iSROA_probability'].iloc[0]    
        # normalize the probabilities by their sum
        pSum = (pCTRL + pISROA)
        pCTRL /= pSum
        pISROA /= pSum
        if np.isnan(pSum):
            print('WARNING: Probabilities for the knee %s missing' % labels['Knee'].iloc[s])
            invalidFlag = True
            continue
#            if labels['iSROA'].iloc[s]==0:
#                iSROA_probability[s] = 1.0
#                estimatedClass[s] = 1
#            else:
#                iSROA_probability[s] = 0.0
#                estimatedClass[s] = 0
        else:
            estimatedClass[s] = np.argmax([pCTRL, pISROA])
            iSROA_probability[s] = pISROA

    if invalidFlag:
        # if at least one knee was missing
        raise ValueError('Submission was incomplete. Please resubmit')    

    bacc = balanced_accuracy_score(labels['iSROA'], estimatedClass)
    rocauc = roc_auc_score(labels['iSROA'], iSROA_probability)

    return bacc, rocauc

if __name__ == "__main__":

    parser.add_argument('--labels', dest='labelFile', help='CSV file containing the ground truth labels '\
    'Needs to be in .csv format')

    parser.add_argument('--predictions', dest='predictionFile', help='CSV file containing the '
    'predictions. Needs to be in the same format as TeamName-Index_Leaderboard_Submission_KNOAP.csv')

    args = parser.parse_args()

    labelFile = args.labelFile
    predictionFile = args.predictionFile

    if not predictionFile.endswith('.csv'):
        raise ValueError('Leaderboard submission filename is not in the correct format: TeamName-Index_Leaderboard_Submission_KNOAP.csv')

    labels = pd.read_csv(labelFile)
    predictions = pd.read_csv(predictionFile)

    bacc, rocauc = calc_metrics(labels, predictions)

    print('########### Metrics ##################')
    print('ROC AUC', rocauc)
    print('BACC', bacc)

    print('\n\n########### File is ready for leaderboard submission to KNOAP ###########')
