import copy
import gc
import multiprocessing as mp

from AsyncDatasetLoader import AsyncCudaDataLoader
import torch
from torch import optim, nn
from tqdm import tqdm

from instances import logger
from utils import compute_accuracy, get_dataloader


def train_net_scaffold(args, net_id, net, global_model, c_local, c_global, dataidxs, test_dataloader, shareIdxs, epochs,
                       lr, args_optimizer, device="cpu", iidEnhance=0):
    shareDL = None
    noise_level = args.noise
    if net_id == args.n_parties - 1:
        noise_level = 0

    if args.noise_type == 'space':
        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                             dataidxs, noise_level, net_id, args.n_parties - 1)
        if len(shareIdxs) != 0:
            shareDL, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                              shareIdxs, noise_level, net_id, args.n_parties - 1)

    else:
        noise_level = args.noise / (args.n_parties - 1) * net_id
        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                             dataidxs, noise_level)
        if len(shareIdxs) != 0:
            shareDL, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                              shareIdxs, noise_level)

    train_dataloader = AsyncCudaDataLoader(train_dl_local, queue_size=4, name="train_dl_local")
    if test_dataloader is not None:
        test_dataloader = AsyncCudaDataLoader(test_dataloader, device, queue_size=4, name="test_dl")
    if shareDL is not None:
        shareDL = AsyncCudaDataLoader(shareDL, device, queue_size=4, name="shareDL")

    logger.info('Training network %s' % str(net_id))
    train_acc, test_acc = -1, -1

    # train_acc = compute_accuracy(net, train_dataloader, device=device)
    if test_dataloader is not None:
        test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    # writer = SummaryWriter()
    net.to(device)
    c_local.to(device)
    c_global.to(device)
    global_model.to(device)

    c_global_para = c_global.state_dict()
    c_local_para = c_local.state_dict()

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            gradNorm, gradAlloc, dlLen, shareLen, = 1, 1, len(tmp), 0
            if shareDL is not None:
                shareLen = len(shareDL)
                shareIter = shareDL.__iter__()
                gradAlloc = (dlLen - shareLen) / dlLen
            else:
                shareIter = None
            if dlLen != 0 and shareDL is not None:
                gradNorm = shareLen / dlLen

            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()

                if shareDL is not None and iidEnhance > 0:
                    try:
                        shareX, shareY = shareIter.__next__()
                    except StopIteration:
                        shareIter = shareDL.__iter__()
                        shareX, shareY = shareIter.__next__()
                    shareX, shareY = shareX.to(device), shareY.to(device).long()
                    shareOut = net(shareX)
                    shareLoss = iidEnhance * gradNorm * criterion(shareOut, shareY).mean()
                    shareLoss.backward()

                optimizer.step()

                net_para = net.state_dict()
                for key in net_para:
                    net_para[key] = net_para[key] - args.lr * (c_global_para[key] - c_local_para[key])
                net.load_state_dict(net_para)

                cnt += 1
                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    net.to('cpu')
    c_local.to("cpu")

    c_new_para = c_local.state_dict()
    c_delta_para = copy.deepcopy(c_local.state_dict())
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    for key in net_para:
        c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (
                cnt * args.lr)
        c_delta_para[key] = c_new_para[key] - c_local_para[key]
    c_local.load_state_dict(c_new_para)

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    logger.info('>> Training accuracy: %f' % train_acc)
    if test_dataloader is not None:
        test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        logger.info('>> Test accuracy: %f' % test_acc)

    logger.info(' ** Training complete **')

    for tmp in train_dataloader:
        if isinstance(tmp, AsyncCudaDataLoader):
            tmp.terminate()
    if isinstance(test_dataloader, AsyncCudaDataLoader):
        test_dataloader.terminate()
    if isinstance(shareDL, AsyncCudaDataLoader):
        shareDL.terminate()

    gc.collect()
    return train_acc, test_acc, c_delta_para, net, c_local


def local_train_net_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map, shareIdxMap=None,
                             test_dl=None, device="cpu", desc="", gradFix=False, iidEnhance=0, pool=None):
    avg_acc = 0.0
    jobs = []
    if pool is not None:
        assert isinstance(pool, mp.pool.Pool)
    total_delta = copy.deepcopy(global_model.state_dict())
    for key in total_delta:
        total_delta[key] = 0.0
    c_global.to(device)
    global_model.to(device)
    if desc != "" and pool is None:
        iterObj = tqdm(nets.items(), desc=desc)
    else:
        iterObj = nets.items()

    if iidEnhance is not None:
        if iidEnhance <= 0:
            iidEnhance = 0

    for net_id, net in iterObj:
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]
        if gradFix and shareIdxMap is not None:
            shareIdxs = shareIdxMap[net_id]
        else:
            shareIdxs = []
        if shareIdxMap is not None:
            dataidxs = dataidxs + shareIdxMap[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to("cpu")
        c_nets[net_id].to("cpu")

        n_epoch = args.epochs

        if pool is None:
            trainacc, testacc, c_delta_para, _, _ = train_net_scaffold(args, net_id, net, global_model, c_nets[net_id],
                                                                       c_global, dataidxs, test_dl, shareIdxs,
                                                                       n_epoch, args.lr, args.optimizer, device,
                                                                       iidEnhance)

            c_nets[net_id].to('cpu')
            for key in total_delta:
                total_delta[key] += c_delta_para[key]

            logger.info("net %d final test acc %f" % (net_id, testacc))
            avg_acc += testacc

            net.to("cpu")
            c_nets[net_id].to('cpu')
        else:
            trainFunArgs = (args, net_id, net, global_model, c_nets[net_id],
                            c_global, dataidxs, test_dl, shareIdxs,
                            n_epoch, args.lr, args.optimizer, device,
                            iidEnhance)
            job = pool.apply_async(train_net_scaffold, args=trainFunArgs)
            jobs.append((net_id, job))

    if pool is not None:
        with tqdm(total=len(jobs), desc=desc) as pbar:
            for net_id, job in jobs:
                trainacc, testacc, c_delta_para, net, c_local = job.get()

                nets[net_id] = net
                c_nets[net_id] = c_local
                nets[net_id].to("cpu")
                c_nets[net_id].to('cpu')

                for key in total_delta:
                    total_delta[key] += c_delta_para[key]

                logger.info("net %d final test acc %f" % (net_id, testacc))
                avg_acc += testacc

                pbar.update(1)

    for key in total_delta:
        total_delta[key] /= args.n_parties
    c_global_para = c_global.state_dict()
    for key in c_global_para:
        if c_global_para[key].type() == 'torch.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.LongTensor)
        elif c_global_para[key].type() == 'torch.cuda.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
        else:
            # print(c_global_para[key].type())
            c_global_para[key] += total_delta[key]
    c_global.load_state_dict(c_global_para)

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    gc.collect()
    return nets, c_nets, c_global
