import os
import multiprocessing as mp
from AsyncDatasetLoader import AsyncCudaDataLoader
import gc
from poolTools import initPool, setSelf
from tqdm import tqdm

proxyAddr = os.environ.get("proxy") if os.environ.get("proxy") else None
if isinstance(proxyAddr, str):
    os.environ["http_proxy"] = proxyAddr
    os.environ["https_proxy"] = proxyAddr

import random

from dataTools.partition import partition_data
from fedAlgo.fedavg import local_train_net
from instances import *
from models.init_nets import init_nets
from dataShare.sharePolicy import uncertainShare, maxLossShare, getIdx, getOtherNet, ntkShare, el2nShare, \
    maxLossEl2nShare
from utils import *
from models.resnetcifar import *


def detaWSave(saveFile, gOld, gNewDict, nets):
    clientNetVecs = {
        netIdx: torch.cat([
            lPara.detach().cpu().view(-1)
            for lPara in nets[netIdx].parameters()]
        ).cpu().numpy()
        for netIdx in nets
    }

    gOldVec = torch.cat([
        para.detach().cpu().view(-1)
        for para in gOld.parameters()]
    ).cpu().numpy()

    gNewVec = torch.cat([
        gNewDict[paraName].detach().cpu().view(-1)
        for paraName in gNewDict]
    ).cpu().numpy()

    wLogData = {
        "gOld": gOldVec,
        "gNew": gNewVec,
        "client": clientNetVecs
    }
    dumpObj(saveFile, wLogData)


def selShares(args, device, nets, net_dataidx_map, shareLog):
    netShareIdx = {}
    for netIdx in nets:
        dataIdxs = net_dataidx_map[netIdx]
        net = nets[netIdx]
        # net.load_state_dict(getOtherNet(net, global_model, fed_avg_freqs[idx]))
        if len(dataIdxs) == 0:
            netShareIdx[netIdx] = []
            continue
        if args.sharePolicy == "random":
            randIdx = np.random.permutation(dataIdxs)
            shareIdx = getIdx(randIdx, args.numShare, np.array(shareLog[netIdx]))
        elif args.sharePolicy == "uncertain":
            shareIdx = uncertainShare(net, dataIdxs, device, args, np.array(shareLog[netIdx]))
        elif args.sharePolicy == "maxLoss":
            shareIdx = maxLossShare(net, dataIdxs, device, args, np.array(shareLog[netIdx]))
        elif args.sharePolicy == "ntkShare":
            shareIdx = ntkShare(net, dataIdxs, device, args, np.array(shareLog[netIdx]))
        elif args.sharePolicy == "el2nShare":
            shareIdx = el2nShare(net, dataIdxs, device, args, np.array(shareLog[netIdx]))
        elif args.sharePolicy == "maxLossEl2nShare":
            shareIdx = maxLossEl2nShare(net, dataIdxs, device, args, np.array(shareLog[netIdx]))
        else:
            shareIdx = []
        netShareIdx[netIdx] = shareIdx
    return netShareIdx


def run(args, pool=None):
    arg2Log = json.dumps(args.__dict__, indent=4) + "\n"
    print(arg2Log)
    mkdirs(args.resDir)
    if isinstance(args.weightLog, str):
        mkdirs(args.weightLog)
    resPath = os.path.join(args.resDir, args.resFile)
    with open(resPath, "w", encoding="utf-8") as f:
        f.write("-" * 20 + "Args" + "-" * 20 + "\n")
        f.write(arg2Log)
        f.write("-" * 20 + "Training" + "-" * 20 + "\n")
        f.write("comm_round\tGlobalTrainAcc\tGlobalTestAcc\n")

    device = torch.device(args.device)
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)

    n_classes = len(np.unique(y_train))

    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                      args.datadir,
                                                                                      args.glob_eval_bs,
                                                                                      args.glob_eval_bs)

    print("len train_dl_global:", len(train_ds_global))

    data_size = len(test_ds_global)

    # test_dl = data.DataLoader(dataset=test_ds_global, batch_size=32, shuffle=False)

    train_all_in_list = []
    test_all_in_list = []
    if args.noise > 0:
        for party_id in range(args.n_parties):
            dataidxs = net_dataidx_map[party_id]

            noise_level = args.noise
            if party_id == args.n_parties - 1:
                noise_level = 0

            if args.noise_type == 'space':
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset,
                                                                                              args.datadir,
                                                                                              args.batch_size, 32,
                                                                                              dataidxs, noise_level,
                                                                                              party_id,
                                                                                              args.n_parties - 1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * party_id
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset,
                                                                                              args.datadir,
                                                                                              args.batch_size, 32,
                                                                                              dataidxs, noise_level)
            train_all_in_list.append(train_ds_local)
            test_all_in_list.append(test_ds_local)
        train_all_in_ds = data.ConcatDataset(train_all_in_list)
        train_dl_global = data.DataLoader(dataset=train_all_in_ds, batch_size=args.glob_eval_bs, shuffle=True)
        test_all_in_ds = data.ConcatDataset(test_all_in_list)
        test_dl_global = data.DataLoader(dataset=test_all_in_ds, batch_size=args.glob_eval_bs, shuffle=False)

    logger.info("Initializing nets")
    nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
    global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
    global_model = global_models[0]
    train_dl_global = AsyncCudaDataLoader(train_dl_global, queue_size=4, name="EvalTrainDL")
    test_dl_global = AsyncCudaDataLoader(test_dl_global, queue_size=4, name="EvalTestDL")

    global_para = global_model.state_dict()
    if args.is_same_initial:
        for net_id, net in nets.items():
            net.load_state_dict(global_para)
    resF = open(resPath, "a", encoding="utf-8")
    shareLog = {i: [] for i in range(len(nets))}
    net_dataidx_map = {x: list(net_dataidx_map[x]) for x in net_dataidx_map}
    loss = nn.CrossEntropyLoss(reduction='none').to(device)

    for round in range(args.comm_round):
        logger.info("in comm round:" + str(round))

        arr = np.arange(args.n_parties)
        np.random.shuffle(arr)
        selected = arr[:int(args.n_parties * args.sample)]

        global_para = global_model.state_dict()
        if round == 0:
            if args.is_same_initial:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)
        else:
            for idx in selected:
                nets[idx].load_state_dict(global_para)

        if args.numShare > 0:
            desc = "{} round:{} {}".format(args.resFile, round, "Data Share")

            netsCut = split_dict_by_count(nets, args.pSize)
            if pool:
                jobs = []
                for netIdx in nets:
                    nets[netIdx].to("cpu")
                for cut in netsCut:
                    shareArgs = (args, device, cut, net_dataidx_map, shareLog)
                    job = pool.apply_async(selShares, args=shareArgs)
                    jobs.append(job)
                with tqdm(total=len(nets), desc=desc) as pbar:
                    for job in jobs:
                        netShareIdx = job.get()
                        for netIdx in netShareIdx:
                            shareIdx = netShareIdx[netIdx]

                            for sidx in shareIdx:
                                net_dataidx_map[netIdx].remove(sidx)
                            shareLog[netIdx].extend(shareIdx)

                            selPartShare = np.delete(np.arange(0, len(nets)), netIdx)
                            np.random.shuffle(selPartShare)
                            selPartShare = selPartShare[:int(len(nets) * args.shareRate)]
                            for otherNetIdx in selPartShare:
                                shareLog[otherNetIdx].extend(shareIdx)

                        pbar.update(args.pSize)
            else:
                with tqdm(total=len(nets), desc=desc) as pbar:
                    for cut in netsCut:
                        netShareIdx = selShares(args, device, cut, net_dataidx_map, shareLog)
                        for netIdx in netShareIdx:
                            shareIdx = netShareIdx[netIdx]

                            for sidx in shareIdx:
                                net_dataidx_map[netIdx].remove(sidx)
                            shareLog[netIdx].extend(shareIdx)

                            selPartShare = np.delete(np.arange(0, len(nets)), netIdx)
                            np.random.shuffle(selPartShare)
                            selPartShare = selPartShare[:int(len(nets) * args.shareRate)]
                            for otherNetIdx in selPartShare:
                                shareLog[otherNetIdx].extend(shareIdx)
                        pbar.update(args.pSize)

        desc = "{} round:{} {}".format(args.resFile, round, "Local Train")
        if args.iidEnhance is None or args.iidEnhance < 0:
            iidEnhance = args.sigma * ((-1 / args.miu) * round + 1)
        else:
            iidEnhance = args.iidEnhance
        if iidEnhance < 0:
            iidEnhance = 0
        if not args.gradFix:
            iidEnhance = 0
        if pool:
            netsCut = split_dict_by_count(nets, args.pSize)
            jobs = []
            for netIdx in nets:
                nets[netIdx].to("cpu")
            for cut in netsCut:
                localTrainArgs = (cut, selected, args, net_dataidx_map, shareLog, None, device, "",
                                  args.gradFix, iidEnhance)
                job = pool.apply_async(local_train_net, args=localTrainArgs)
                jobs.append(job)
            with tqdm(total=len(nets), desc=desc) as pbar:
                for job in jobs:
                    nets.update(job.get())
                    pbar.update(args.pSize)
        else:
            nets = local_train_net(nets, selected, args, net_dataidx_map, shareLog, None, device, desc,
                                   args.gradFix, iidEnhance)

        # local_train_net(nets, selected, args, net_dataidx_map, test_dl=test_dl_global, device=device,
        #                expID=expID, round=round)
        # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

        # update global model
        # 聚合权重修复，这是导致Cifar10上训练到后期劣化的原因
        total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

        for idx in range(len(selected)):
            net_para = nets[selected[idx]].cpu().state_dict()
            if idx == 0:
                for key in net_para:
                    global_para[key] = net_para[key] * fed_avg_freqs[idx]
            else:
                for key in net_para:
                    global_para[key] += net_para[key] * fed_avg_freqs[idx]

        if isinstance(args.weightLog, str):
            savePath = os.path.join(args.weightLog, "{}_round.pickle".format(round))
            detaWSave(savePath, global_model, global_para, nets)
        global_model.load_state_dict(global_para)

        logger.info('global n_training: %d' % len(train_dl_global))
        logger.info('global n_test: %d' % len(test_dl_global))

        global_model.to(args.evalDev)
        train_acc = compute_accuracy(global_model, train_dl_global, device=args.evalDev)
        test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True,
                                                 device=args.evalDev)
        global_model.to(device)

        logger.info('>> Global Model Train accuracy: %f' % train_acc)
        logger.info('>> Global Model Test accuracy: %f' % test_acc)
        # print("round:{}\ttrain_acc:{}\ttest_acc:{}".format(round, train_acc, test_acc))
        resF.write("{}\t{}\t{}\n".format(round, train_acc, test_acc))
        resF.flush()

    train_dl_global.terminate()
    test_dl_global.terminate()
    del train_dl_global
    del test_dl_global
    gc.collect()


if __name__ == '__main__':
    # torch.set_printoptions(profile="full")

    """
参数：
--model=simple-cnn
--dataset=cifar10
--alg=fedavg
--lr=0.01
--batch-size=64
--epochs=10
--n_parties=10
--rho=0.9
--comm_round=50
--partition=noniid-labeldir
--beta=0.5
--device=cuda:0
--datadir=./data/
--logdir=./logs/
--noise=0.0
--init_seed=0
    """

    args.model = "mlp"  # 下文接管
    args.dataset = "covtype"  # 下文接管
    args.alg = "fedavg"
    args.lr = 0.01
    args.batch_size = 64
    args.glob_eval_bs = 10000
    args.epochs = 10  # 下文接管
    args.n_parties = 10
    args.rho = 0.9
    args.comm_round = 50  # 下文接管
    args.partition = "noniid-#label1"  # 下文接管
    args.beta = 0.5
    args.device = 'cpu'
    args.evalDev = "cuda:0"
    args.datadir = 'R:/data/'
    args.logdir = './logs/'
    args.noise = 0.0
    args.init_seed = 0
    args.numShare = 2  # 下文接管
    args.sharePolicy = "random"  # 下文接管
    args.download = False
    args.resFile = "None-0_share.tab"  # 下文接管
    args.resDir = "./Results"  # 下文接管
    args.shareRate = 0.05  # 下文接管
    args.pSize = 1
    args.gradFix = True
    args.miu = 15
    args.sigma = 2
    args.iidEnhance = None  # 下文接管
    args.weightLog = None  # 下文接管

    poolSize = 10
    cpuSets = None
    NoShareExp = False
    useWeightLog = False
    datasetList = [
        ("covtype", "mlp"),
        ("rcv1", "mlp"),
        ("cifar10", "simple-cnn"),
        ("svhn", "simple-cnn")
    ]
    totalEPList = [450, 600, 750]
    locEpList = [5, 10, 15]
    miuList = [5, 15, 30, 50]
    sigmaList = [1, 2, 4, 8]
    iidEnhanceList = [0]
    shareList = [2]
    sharePolicyList = ["random"]
    # sharePolicyList = ["random", "uncertain", "maxLoss", "el2nShare"]
    partitionList = ["noniid-#label1"]
    # partitionList = ["noniid-labeldir", "noniid-#label1", "noniid-#label1", "homo"]
    shareRateList = [1]
    tabDir = "./Results"

    if args.gradFix or args.iidEnhance == 0:
        tabDir = tabDir + "-gradFix"
    mp.set_start_method("spawn")
    if not isinstance(poolSize, int) or poolSize < 0:
        poolSize = min(int(args.n_parties / args.pSize), mp.cpu_count())
    if poolSize != 0:
        if not cpuSets:
            cpuSets = [i for i in range(0, mp.cpu_count())]
        setSelf(cpuList=cpuSets)
        pool = initPool(poolSize, cpuList=[cpuSets for _ in range(poolSize)])
    else:
        pool = None
    mkdirs(tabDir)
    runningStatusFile = open(os.path.join(tabDir, "running_status.log"), "w", encoding="utf-8")

    miuSigma = []
    for miu in miuList:
        for sigma in sigmaList:
            miuSigma.append((miu, sigma, None))
    for iidE in iidEnhanceList:
        miuSigma.append((1, 0, iidE))

    # 无数据交换
    if NoShareExp:
        for totalEP in totalEPList:
            for locEp in locEpList:
                for dataset, netName in datasetList:
                    for part in partitionList:
                        commRound = int(totalEP / locEp)

                        args.model = netName
                        args.dataset = dataset
                        args.epochs = locEp
                        args.comm_round = commRound
                        args.shareRate = 0
                        args.sharePolicy = None
                        args.partition = part
                        args.numShare = 0
                        args.resDir = os.path.join(tabDir, "epochs=[{}]_totalRound=[{}]".format(locEp, commRound),
                                                   dataset, part)
                        args.resFile = "None-0_share.tab"
                        if useWeightLog:
                            args.weightLog = os.path.join(args.resDir, "None-0_share-wLogDir")
                        else:
                            args.weightLog = None

                        run(args, pool)

    # 根据参数排列组合数据交换
    outDirs = set()
    for totalEP in totalEPList:
        for locEp in locEpList:
            for dataset, netName in datasetList:
                for part in partitionList:
                    for sp in sharePolicyList:
                        for i in range(0, len(shareList)):
                            for sr in shareRateList:
                                for miu, sigma, iidE in miuSigma:
                                    commRound = int(totalEP / locEp)

                                    args.model = netName
                                    args.dataset = dataset
                                    args.epochs = locEp
                                    args.comm_round = commRound
                                    args.shareRate = sr
                                    args.sharePolicy = sp
                                    args.partition = part
                                    args.numShare = shareList[i]
                                    args.miu = miu
                                    args.sigma = sigma
                                    args.iidEnhance = iidE
                                    args.resDir = os.path.join(tabDir,
                                                               "epochs=[{}]_totalRound=[{}]".format(locEp, commRound),
                                                               dataset, part, "rate_{}".format(sr))
                                    args.resFile = "{}-{}_share_miu={}_sigma={}_iidE={}.tab".format(args.sharePolicy,
                                                                                                    shareList[i],
                                                                                                    miu, sigma, iidE)
                                    outDirs.add(args.resDir)
                                    if useWeightLog:
                                        wlogDir = "{}-{}_share-wLogDir".format(args.sharePolicy, shareList[i])
                                        args.weightLog = os.path.join(args.resDir, wlogDir)
                                    else:
                                        args.weightLog = None

                                    run(args, pool)

    if isinstance(pool, mp.pool.Pool):
        pool.close()
        pool.join()
        pool.terminate()

    # from ResultParser import dir2Excel
    #
    # if NoShareExp:
    #     for part in partitionList:
    #         baseTab = os.path.join(tabDir, part, "None-0_share.tab")
    #         for dataDir in outDirs:
    #             copyfile(baseTab, os.path.join(dataDir, "None-0_share.tab"))
    # for dataDir in outDirs:
    #     excelName = "summary.xlsx"
    #     outPath = os.path.join(dataDir, excelName)
    #     dir2Excel(dataDir, outPath)
    #     if NoShareExp:
    #         os.remove(os.path.join(dataDir, "None-0_share.tab"))
    #     print("Res excel: {}".format(outPath))
