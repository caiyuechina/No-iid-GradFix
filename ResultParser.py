import pandas as pd
import json
import os


def tabLoad(file):
    with open(file, 'r', encoding="utf-8") as f:
        tabs = f.read()
    return tabLoads(tabs)


def tabLoads(txt):
    rows = list(filter(lambda row: len(row) > 0, txt.split('\n')))
    mateIdx, tableIdx = 0, -1
    for i in range(1, len(rows)):
        if "-" * 20 in rows[i]:
            tableIdx = i
            break
    if tableIdx == -1:
        raise Exception("table not found")
    mateData = json.loads("\n".join(rows[mateIdx + 1:tableIdx]))
    tableData = [rows[i].split("\t") for i in range(tableIdx + 1, len(rows))]
    tableData = pd.DataFrame(tableData[1:], columns=tableData[0])
    return mateData, tableData.astype('float')


def colRename(tabFileDatas):
    for mateData, tableData in tabFileDatas:
        sharePolicy = mateData["sharePolicy"] if mateData["sharePolicy"] else None
        tableData["comm_round"] = tableData["comm_round"].astype('int')
        reNameMap = {
            "comm_round": "{}-comm_round".format(sharePolicy),
            "GlobalTrainAcc": "{}-TrainAcc".format(sharePolicy),
            "GlobalTestAcc": "{}-TestAcc".format(sharePolicy)
        }
        tableData.rename(columns=reNameMap, inplace=True)


def dir2Excel(dataDir, outExcel):
    tabFileList = list(filter(lambda fName: fName.endswith(".tab"), os.listdir(dataDir)))
    tabFileList = [os.path.join(dataDir, fName) for fName in tabFileList]
    tabFileDatas = [tabLoad(fPath) for fPath in tabFileList]

    noIID = tabFileDatas[0][0]["partition"]
    colRename(tabFileDatas)

    sheets = []
    noneShareDF = None
    numShareList = sorted(set([tdata[0]["numShare"] for tdata in tabFileDatas]))
    if 0 in numShareList:
        noneShareDF = filter(lambda tdata: tdata[0]["numShare"] == 0, tabFileDatas).__next__()[1]
        numShareList.remove(0)
    for numShare in numShareList:
        tmpData = list(filter(lambda tdata: tdata[0]["numShare"] == numShare, tabFileDatas))
        tmpData = sorted(tmpData, key=lambda tdata: tdata[0]["sharePolicy"])
        if isinstance(noneShareDF, pd.DataFrame):
            sheetData = pd.concat((noneShareDF, *[tData[1] for tData in tmpData]), axis=1)
        else:
            sheetData = pd.concat([tData[1] for tData in tmpData], axis=1)
        sheetName = "{}-share{}".format(noIID, numShare)
        sheets.append((sheetName, sheetData))

    with pd.ExcelWriter(outExcel, engine="xlsxwriter") as writer:
        for sheetName, sheetData in sheets:
            sheetData.to_excel(writer, sheet_name=sheetName, index=False)


if __name__ == '__main__':
    dataDir = r"E:\项目\NIID-Bench\Results\noniid-labeldir"
    outExcel = r"E:\项目\NIID-Bench\Results\noniid-labeldir\svhn-noniid-labeldir.xlsx"
    dir2Excel(dataDir, outExcel)
    print("Finish output: {}".format(outExcel))
