import pandas as pd
import numpy as np


def featGen(inputDict, binSize, aggCategory, nonAggCategory):
    listAllInstances = inputDict['dataAll']
    binFeatPerSession = []
    binLabelPerSession = []

    for ins in range(len(listAllInstances)):
        listPerInstance = listAllInstances[ins]
        binFeat = pd.DataFrame()
        binLabel = pd.DataFrame()
        accDataReceived = False
        for dataSource in range(len(listPerInstance)):
            df = listPerInstance[dataSource]
            df['Condition'].fillna(0, inplace=True)  # fill NaN with 0
            df['Condition'][df['Condition'].isin(aggCategory)] = 1   # 1 to Agg state
            df['Condition'][df['Condition'].isin(nonAggCategory)] = 0  # 0 to nonAgg state

            # Adding the norm of accelerometer data to the data frame.
            # if not accDataReceived:
            if 'X' in df:
                acc_data = (df[['X', 'Y', 'Z']]).to_numpy().astype(np.float)
                df['ACCNorm'] = np.linalg.norm(acc_data, axis=1)

            if df['Condition'].nunique() > 2:
                print(df['Condition'].nunique())
                assert ('Have unknown labels!')

            evidence = list(df.columns.values)
            evidence.remove('Condition')
            if len(evidence) > 1:
                accDataReceived = True
            else:
                pass
            for e in range(len(evidence)):
                binFeat[evidence[e]+'first'] = pd.to_numeric(df[evidence[e]].dropna()).resample(binSize).first()
                binFeat[evidence[e]+'last'] = pd.to_numeric(df[evidence[e]].dropna()).resample(binSize).last()
                binFeat[evidence[e]+'max'] = pd.to_numeric(df[evidence[e]].dropna()).resample(binSize).max()
                binFeat[evidence[e]+'min'] = pd.to_numeric(df[evidence[e]].dropna()).resample(binSize).min()
                binFeat[evidence[e]+'mean'] = pd.to_numeric(df[evidence[e]].dropna()).resample(binSize).mean()
                binFeat[evidence[e]+'median'] = pd.to_numeric(df[evidence[e]].dropna()).resample(binSize).median()
                binFeat[evidence[e]+'nunique'] = pd.to_numeric(df[evidence[e]].dropna()).resample(binSize).nunique()
                binFeat[evidence[e]+'std'] = pd.to_numeric(df[evidence[e]].dropna()).resample(binSize).std()
                binFeat[evidence[e]+'sum'] = pd.to_numeric(df[evidence[e]].dropna()).resample(binSize).sum()
                binFeat[evidence[e]+'var'] = pd.to_numeric(df[evidence[e]].dropna()).resample(binSize).var()
                binLabel[evidence[e]+'Label'] = df.Condition.resample(binSize).max()

        binLabel = binLabel.apply(lambda x: x.dropna().max(), axis=1)

        # Add elapsed time to/till aggression features
        if np.sum(binLabel) == 0:
            #no Aggression obeserved!
            binFeat['AGGObserved'] = 0
            binFeat['TimePastAggression'] = 50000000
        else:
            aggObservedFeat = binLabel.copy()
            timePastAggFeat = binLabel.copy()
            aggInd = np.where(binLabel != 0)[0] # index of agression Labels
            minAgg = np.min(aggInd)  # get index of first Agg episode
            counter = 0
            for t in range(0, len(binLabel)):
                if t >= minAgg:
                    aggObservedFeat[t] = 1
                    if t in aggInd:
                        counter = 0
                    else:
                        pass
                    timePastAggFeat[t] = counter
                    counter = counter + 1
                else:
                    aggObservedFeat[t] = 0
                    timePastAggFeat[t] = 50000000
                    
            #adding new time since last agg features
            binFeat['AGGObserved'] = aggObservedFeat
            binFeat['TimePastAggression'] = timePastAggFeat

        if not accDataReceived:
            for ax in ['X', 'Y', 'Z']:
                binFeat[ax+'first'] = 0
                binFeat[ax+'last'] = 0
                binFeat[ax+'max'] = 0
                binFeat[ax+'min'] = 0
                binFeat[ax+'mean'] = 0
                binFeat[ax+'median'] = 0
                binFeat[ax+'nunique'] = 0
                binFeat[ax+'std'] = 0
                binFeat[ax+'sum'] = 0
                binFeat[ax+'var'] = 0
        else:
            properOrderOfFeats = binFeat.columns

        binFeatPerSession.append(binFeat[properOrderOfFeats])
        binLabelPerSession.append(binLabel)

    outputDict = {'features': binFeatPerSession, 'labels': binLabelPerSession}
    return outputDict
