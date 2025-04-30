import gc

import torch
from torch import optim, nn
from tqdm import tqdm
import multiprocessing as mp

from AsyncDatasetLoader import AsyncCudaDataLoader
from instances import logger
from utils import compute_accuracy, get_dataloader


def train_net_fedprox(args, net_id, net, global_net, dataidxs, test_dataloader, shareIdxs, epochs, lr, args_optimizer,
                      mu, device="cpu", iidEnhance=0):
    net.to(device)
    shareDL = None
    noise_level = args.noise
    if net_id == args.n_parties - 1:
        noise_level = 0

    if args.noise_type == 'space':
        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                             dataidxs, noise_level, net_id, args.n_parties - 1)
        if len(shareIdxs) != 0:
            shareDL, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, shareIdxs,
                                              noise_level, net_id, args.n_parties - 1)
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
    # mu = 0.001
    global_weight_collector = list(global_net.to(device).parameters())

    for epoch in range(epochs):
        epoch_loss_collector = []

        gradNorm, gradAlloc, dlLen, shareLen, = 1, 1, len(train_dataloader), 0
        if shareDL is not None:
            shareLen = len(shareDL)
            shareIter = shareDL.__iter__()
            gradAlloc = (dlLen - shareLen) / dlLen
        else:
            shareIter = None
        if dlLen != 0 and shareDL is not None:
            gradNorm = shareLen / dlLen

        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = net(x)
            loss = criterion(out, target)

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

            # for fedprox
            fed_prox_reg = 0.0
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
            loss += fed_prox_reg

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        # if epoch % 10 == 0:
        #     train_acc = compute_accuracy(net, train_dataloader, device=device)
        #     test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        #
        #     logger.info('>> Training accuracy: %f' % train_acc)
        #     logger.info('>> Test accuracy: %f' % test_acc)

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    logger.info('>> Training accuracy: %f' % train_acc)
    if test_dataloader is not None:
        test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        logger.info('>> Test accuracy: %f' % test_acc)

    net.to('cpu')
    logger.info(' ** Training complete **')

    if isinstance(train_dataloader, AsyncCudaDataLoader):
        train_dataloader.terminate()
    if isinstance(test_dataloader, AsyncCudaDataLoader):
        test_dataloader.terminate()
    if isinstance(shareDL, AsyncCudaDataLoader):
        shareDL.terminate()

    gc.collect()
    return train_acc, test_acc, net


def local_train_net_fedprox(nets, selected, global_model, args, net_dataidx_map, shareIdxMap=None,
                            test_dl=None, device="cpu", desc="", gradFix=False, iidEnhance=0, pool=None):
    avg_acc = 0.0
    jobs = []
    if pool is not None:
        assert isinstance(pool, mp.pool.Pool)
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
        net.to(device)

        n_epoch = args.epochs
        if pool is not None:
            trainFunArgs = (args, net_id, net, global_model, dataidxs, test_dl, shareIdxs,
                            n_epoch, args.lr, args.optimizer, args.mu, device, iidEnhance)
            job = pool.apply_async(train_net_fedprox, args=trainFunArgs)
            jobs.append((net_id, job))
        else:
            trainacc, testacc, net = train_net_fedprox(args, net_id, net, global_model, dataidxs, test_dl, shareIdxs,
                                                       n_epoch, args.lr, args.optimizer, args.mu, device, iidEnhance)
            net.to('cpu')
            nets[net_id] = net

            logger.info("net %d final test acc %f" % (net_id, testacc))
            avg_acc += testacc

    if pool is not None:
        with tqdm(total=len(jobs), desc=desc) as pbar:
            for net_id, job in jobs:
                trainacc, testacc, net = job.get()

                net.to('cpu')
                nets[net_id] = net

                logger.info("net %d final test acc %f" % (net_id, testacc))
                avg_acc += testacc

                pbar.update(1)

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    gc.collect()
    return nets
