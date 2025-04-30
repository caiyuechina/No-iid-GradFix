import gc

from AsyncDatasetLoader import AsyncCudaDataLoader
import torch
from torch import optim, nn, Tensor
from tqdm import tqdm
from instances import logger
from typing import List, Dict, Tuple
from utils import compute_accuracy, get_dataloader
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype


def calcGradNorm(parameters, norm_type):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return torch.tensor(0.)
    first_device = grads[0].device
    grouped_grads: Dict[Tuple[torch.device, torch.dtype], Tuple[List[List[Tensor]], List[int]]] \
        = _group_tensors_by_device_and_dtype([grads])  # type: ignore[assignment]
    norms: List[Tensor] = []
    for ((device, _), ([device_grads], _)) in grouped_grads.items():  # type: ignore[assignment]
        norms.extend([torch.linalg.vector_norm(g, norm_type) for g in device_grads])

    total_norm = torch.linalg.vector_norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)
    return float(total_norm)


def train_net(args, net_id, net, train_dataloader, test_dataloader, shareDL, epochs, lr, args_optimizer, device="cpu",
              iidEnhance=0):
    logger.info('Training network %s' % str(net_id))
    train_acc, test_acc = -1, -1

    # train_acc = compute_accuracy(net, train_dataloader, device=device)
    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    if test_dataloader is not None:
        test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
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
                x, target = x.to(device), target.to(device).long()

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False

                out = net(x)
                loss = criterion(out, target)
                loss.backward()
                # localGradNorm = calcGradNorm(net.parameters(), 2)

                cnt += 1
                epoch_loss_collector.append(loss.item())

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

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        # train_acc = compute_accuracy(net, train_dataloader, device=device)
        # test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

        # writer.add_scalar('Accuracy/train', train_acc, epoch)
        # writer.add_scalar('Accuracy/test', test_acc, epoch)

        # if epoch % 10 == 0:
        #     logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
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
    return train_acc, test_acc


def local_train_net(nets, selected, args, net_dataidx_map, shareIdxMap, test_dl=None, device="cpu", desc="",
                    gradFix=False, iidEnhance=None):
    avg_acc = 0.0
    if desc != "":
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
        if gradFix:
            shareIdxs = shareIdxMap[net_id]
        else:
            shareIdxs = []
        dataidxs = dataidxs + shareIdxMap[net_id]
        shareDL = None

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level, net_id, args.n_parties - 1,
                                                                 download=args.download)
            if len(shareIdxs) != 0:
                shareDL, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                  shareIdxs, noise_level, net_id, args.n_parties - 1,
                                                  download=args.download)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level, download=args.download)
            if len(shareIdxs) != 0:
                shareDL, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                  shareIdxs, noise_level, download=args.download)
        # train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        train_dl_local = AsyncCudaDataLoader(train_dl_local, queue_size=4)
        if test_dl is not None:
            test_dl = AsyncCudaDataLoader(test_dl, device, queue_size=4)
        if shareDL is not None:
            shareDL = AsyncCudaDataLoader(shareDL, device, queue_size=4)

        trainacc, testacc = train_net(args, net_id, net, train_dl_local, test_dl, shareDL, n_epoch, args.lr,
                                      args.optimizer, device, iidEnhance)

        train_dl_local.terminate()
        if test_dl is not None:
            test_dl.terminate()
        if shareDL is not None:
            shareDL.terminate()

        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        # saving the trained models here
        # save_model(net, net_id, args)
        # else:
        #     load_model(net, net_id, device=device)
        net.to('cpu')
    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    gc.collect()
    return nets
