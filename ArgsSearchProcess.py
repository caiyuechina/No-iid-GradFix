import os
import re
from copy import deepcopy

import numpy as np
import pandas as pd

from ResultParser import tabLoad


def parse_tab_data(file_path, rounds):
    # Use your tabLoad function to parse metadata and DataFrame
    try:
        mateData, table_df = tabLoad(file_path)
    except Exception as e:
        print(f"Error parsing file: {file_path}")
        print(f"Error message: {e}")
        return None  # Skip this file if parsing fails

    # Extract desired rounds' GlobalTestAcc values with 0-based indexing
    test_acc = {}
    for r in rounds:
        try:
            test_acc[f"{r}"] = table_df.loc[table_df['comm_round'] == r - 1, 'GlobalTestAcc'].values[0]
        except IndexError:
            test_acc[f"{r}"] = None  # Handle missing rounds gracefully

    # Apply conditions for miu, sigma, and iidEnhance
    iidEnhance_value = mateData.get("iidEnhance", None)
    if iidEnhance_value is not None and iidEnhance_value >= 0:
        miu, sigma = "-", "-"
    else:
        iidEnhance_value = "-"
        miu, sigma = mateData.get("miu", "-"), mateData.get("sigma", "-")
    metaInfo = {
        "dataset": mateData.get("dataset", "-"),
        "miu": miu,
        "sigma": sigma,
        "iidEnhance": iidEnhance_value,
    }
    return metaInfo, test_acc


def aggregate_tab_data(directories, rounds, output_file="aggregated_results.xlsx"):
    all_data, part_data = [], []
    for directory in directories:
        for file in os.listdir(directory):
            if file.endswith(".tab"):
                file_path = os.path.join(directory, file)
                metaInfo, test_acc = parse_tab_data(file_path, rounds)
                part_data.append((metaInfo, np.array([test_acc[key] for key in test_acc])))
                parsed_data = deepcopy(metaInfo)
                parsed_data.update(test_acc)
                if parsed_data:  # Only add valid data
                    all_data.append(parsed_data)

    # Create final DataFrame
    cols = ["dataset", "miu", "sigma", "iidEnhance"] + ["{}".format(i) for i in rounds]
    df = pd.DataFrame(all_data, columns=cols)

    # Sort by dataset, miu, sigma, iidEnhance, with "-" before numeric values in iidEnhance
    df['miu'] = pd.to_numeric(df['miu'], errors='coerce')
    df['sigma'] = pd.to_numeric(df['sigma'], errors='coerce')
    df['iidEnhance_numeric'] = pd.to_numeric(df['iidEnhance'], errors='coerce')  # Temporary column for sorting
    df = df.sort_values(by=['dataset', 'miu', 'sigma', 'iidEnhance_numeric', 'iidEnhance'],
                        key=lambda col: (col == "-") | pd.isna(col) if col.name == 'iidEnhance' else col,
                        ascending=[True, True, True, True, True])

    # Drop the temporary column and reset the index
    df = df.drop(columns='iidEnhance_numeric').reset_index(drop=True)

    # Save to Excel
    df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")
    return part_data


def domainNameProcess(name):
    pattern = r"epochs=\[(\d+)\]_totalRound=\[(\d+)\]"
    match = re.match(pattern, name)
    epochs, totalRound = None, None
    if match:
        epochs = int(match.group(1))
        totalRound = int(match.group(2))
    else:
        print("No match found.")
    return epochs, totalRound


def summaryDomains(domainData, dataSetList, totalRound, cmpRound):
    beginCmp, endCmp = int(totalRound * cmpRound), int(totalRound * (1 - cmpRound))
    dsPart, dsBaseline, argDSAdv = {ds: [] for ds in dataSetList}, {}, {}
    for metaInfo, test_acc in domainData:
        ds = metaInfo["dataset"]
        if metaInfo["iidEnhance"] == 0:
            dsBaseline[ds] = [metaInfo, test_acc]
        else:
            dsPart[ds].append([metaInfo, test_acc])
    for ds in dsPart:
        dsBaseLine = dsBaseline[ds][1]
        for metaInfo, test_acc in dsPart[ds]:
            argDesc = (int(metaInfo["miu"]), int(metaInfo["sigma"]))
            beginAdv = np.sum(test_acc[:beginCmp] - dsBaseLine[:beginCmp])
            endAdv = np.sum(test_acc[endCmp:] - dsBaseLine[endCmp:])
            sumAdv = beginAdv + endAdv
            avgSumAdv = beginAdv / beginCmp + endAdv / (totalRound - endCmp)
            if argDesc not in argDSAdv:
                argDSAdv[argDesc] = {}
            argDSAdv[argDesc][ds] = (beginAdv, endAdv, sumAdv, avgSumAdv)
    res = []
    for argDesc in argDSAdv:
        argInDS = argDSAdv[argDesc]
        dsAvgSumAdv = sum([argInDS[ds][3] for ds in argInDS])
        res.append([argDesc, dsAvgSumAdv])
    return res


if __name__ == "__main__":
    dataSetList = ["cifar10", "covtype", "rcv1", "svhn"]
    domainDirPatten = r"{}\{}\noniid-#label1\rate_1"
    cmpRound = 0.3
    rootDir = "./广域参数搜索"

    roundDomains = list(filter(lambda x: os.path.isdir(os.path.join(rootDir, x)), os.listdir(rootDir)))
    roundDomains = [(os.path.join(rootDir, dName), *domainNameProcess(dName)) for dName in roundDomains]
    allDomains = {}
    for dPath, epochs, totalRound in roundDomains:
        rounds = [i for i in range(1, totalRound + 1)]
        dirs = [domainDirPatten.format(dPath, d) for d in dataSetList]
        part_data = aggregate_tab_data(dirs, rounds, output_file=f"{dPath}.xlsx")
        dSummary = summaryDomains(part_data, dataSetList, totalRound, cmpRound)
        for argDesc, dsSumAdv in dSummary:
            if argDesc not in allDomains:
                allDomains[argDesc] = []
            allDomains[argDesc].append([epochs, totalRound, dsSumAdv])

    coverted = []
    for argDesc in allDomains:
        miu, sigma = argDesc
        row = {"miu": miu, "sigma": sigma}
        tmp = [("local={}\nagg={}".format(epochs, totalRound), dsSumAdv, epochs * totalRound + epochs)
               for epochs, totalRound, dsSumAdv in allDomains[argDesc]]
        tmp = dict([(row[0], row[1]) for row in sorted(tmp, key=lambda x: x[2])])
        row.update(tmp)
        coverted.append(row)
    df = pd.DataFrame(coverted)
    df = df.sort_values(by=['miu', 'sigma'])
    df.to_excel(os.path.join(rootDir, "allSummary.xlsx"), index=False)
