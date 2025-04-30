import numpy as np
import torch
import numba as nb
from torch import nn

from utils import get_dataloader
from torch.func import functional_call, vmap, vjp, jvp, jacrev
import torch.nn.functional as F


def getOtherNet(myNet, globalNet, fedAvgFreq):
    freqFix = 1 / (1 - fedAvgFreq)
    myNetParams = myNet.cpu().state_dict()
    globalNetParams = globalNet.cpu().state_dict()
    otherNetParams = {
        key: (globalNetParams[key] - fedAvgFreq * myNetParams[key]) * freqFix
        for key in myNetParams
    }
    return otherNetParams


def ntk(net, x1, x2, compute):
    params = {k: v.detach() for k, v in net.named_parameters()}

    def fnet_single(params, x):
        return functional_call(net, params, (x.unsqueeze(0),)).squeeze(0)

    def empirical_ntk_jacobian_contraction(fnet_single, params, x1, x2, compute='full'):
        # Compute J(x1)
        jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
        jac1 = jac1.values()
        jac1 = [j.flatten(2) for j in jac1]

        # Compute J(x2)
        jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
        jac2 = jac2.values()
        jac2 = [j.flatten(2) for j in jac2]

        # Compute J(x1) @ J(x2).T
        einsum_expr = None
        if compute == 'full':
            einsum_expr = 'Naf,Mbf->NMab'
        elif compute == 'trace':
            einsum_expr = 'Naf,Maf->NM'
        elif compute == 'diagonal':
            einsum_expr = 'Naf,Maf->NMa'
        else:
            assert False

        result = torch.stack([torch.einsum(einsum_expr, j1, j2) for j1, j2 in zip(jac1, jac2)])
        result = result.sum(0)
        return result

    result = empirical_ntk_jacobian_contraction(fnet_single, params, x1, x2, compute)
    return result


def ntkShare(net, dataidxs, device, args, exclude=None):
    if exclude is None:
        exclude = []
    if args.numShare <= 0:
        return None
    net.to(device)
    train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, 1024, 32,
                                                         dataidxs, 0.0, download=args.download,
                                                         datasetVal=True)
    train_dl_local_val, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, 1024, 32,
                                                             dataidxs, 0.0, download=args.download,
                                                             datasetVal=True)
    ntkEff = []
    for batch_idx, (x, target) in enumerate(train_dl_local):
        x = x.to(device)
        ntkEffCol = []
        for _, (y, target_y) in enumerate(train_dl_local):
            y = y.to(device)
            result = ntk(net, y, x, 'trace')
            ntkEffCol.append(result)
        ntkEffCol = torch.cat(ntkEffCol, dim=0)
        ntkEff.append(ntkEffCol)
    ntkEff = torch.cat(ntkEff, dim=1)
    ntkEff = ntkEff - torch.diag_embed(torch.diag(ntkEff))
    ntkEff = torch.mean(ntkEff, dim=0)

    localSortedHidx = torch.argsort(ntkEff, dim=0, descending=True).cpu().numpy()
    sortedDataIdxs = np.array(dataidxs)[localSortedHidx]
    shareIdx = getIdx(sortedDataIdxs, args.numShare, exclude)
    if shareIdx is None:
        return []
    else:
        return shareIdx


def el2nShare(net, dataidxs, device, args, exclude=None):
    if exclude is None:
        exclude = []
    if args.numShare <= 0:
        return None
    net.to(device)
    train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, 1024, 32,
                                                         dataidxs, 0.0, download=args.download,
                                                         datasetVal=True)
    el2nVar = []
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(train_dl_local):
            x, target = x.to(device), target.to(device).long()
            out = net(x)
            tmp = torch.softmax(out, dim=-1) - F.one_hot(target, num_classes=out.shape[-1])
            el2nVar.append(torch.norm(tmp, p=2, dim=-1))
    el2nVar = torch.cat(el2nVar, dim=0)
    localSortedHidx = torch.argsort(el2nVar, dim=0, descending=True).cpu().numpy()
    sortedDataIdxs = np.array(dataidxs)[localSortedHidx]
    shareIdx = getIdx(sortedDataIdxs, args.numShare, exclude)
    if shareIdx is None:
        return []
    else:
        return shareIdx


@nb.jit(nopython=True)
def getIdx(target, numSelect, exclude=None):
    selCount = 0
    res = np.zeros(numSelect, dtype=target.dtype)
    for i in range(0, len(target)):
        if exclude is not None and target[i] in exclude:
            continue
        res[selCount] = target[i]
        selCount += 1
        if selCount >= numSelect:
            break
    if selCount > 0:
        res = res[:selCount]
    else:
        res = None
    return res


def uncertainShare(net, dataidxs, device, args, exclude=None):
    if exclude is None:
        exclude = []
    if args.numShare <= 0:
        return None
    net.to(device)
    train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, 1024, 32,
                                                         dataidxs, 0.0, download=args.download,
                                                         datasetVal=True)
    entropy = []
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(train_dl_local):
            x = x.to(device)

            out = net(x)
            h = -torch.log_softmax(out, dim=-1) * torch.softmax(out, dim=-1)
            entropy.append(torch.sum(h, dim=1))
    entropy = torch.cat(entropy, dim=0)
    localSortedHidx = torch.argsort(entropy, dim=0, descending=True).cpu().numpy()
    sortedDataIdxs = np.array(dataidxs)[localSortedHidx]
    shareIdx = getIdx(sortedDataIdxs, args.numShare, exclude)
    if shareIdx is None:
        return []
    else:
        return shareIdx


def maxLossShare(net, dataidxs, device, args, exclude=None):
    loss = nn.CrossEntropyLoss(reduction='none').to(device)
    if exclude is None:
        exclude = []
    if args.numShare <= 0:
        return None
    net.to(device)
    train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, 1024, 32,
                                                         dataidxs, 0.0, download=args.download,
                                                         datasetVal=False)
    lossList = []
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(train_dl_local):
            x, y = x.to(device), target.to(device).long()
            out = net(x)
            l = loss(out, y)
            lossList.append(l)
    loss = torch.cat(lossList, dim=0)
    localSortedIdx = torch.argsort(loss, dim=0, descending=True).cpu().numpy()
    sortedDataIdxs = np.array(dataidxs)[localSortedIdx]
    shareIdx = getIdx(sortedDataIdxs, args.numShare, exclude)
    if shareIdx is None:
        return []
    else:
        return shareIdx


def maxLossEl2nShare(net, dataidxs, device, args, exclude=None):
    loss = nn.CrossEntropyLoss(reduction='none').to(device)
    if exclude is None:
        exclude = []
    if args.numShare <= 0:
        return None
    net.to(device)
    train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, 1024, 32,
                                                         dataidxs, 0.0, download=args.download,
                                                         datasetVal=False)
    lossVar, el2nVar = [], []
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(train_dl_local):
            x, y = x.to(device), target.to(device).long()
            out = net(x)
            l = loss(out, y)
            lossVar.append(l)
            tmp = torch.softmax(out, dim=-1) - F.one_hot(y, num_classes=out.shape[-1])
            el2nVar.append(torch.norm(tmp, p=2, dim=-1))
    lossVar = torch.cat(lossVar, dim=0)
    el2nVar = torch.cat(el2nVar, dim=0)
    lossVar = (lossVar - lossVar.mean()) / lossVar.std()
    el2nVar = (el2nVar - el2nVar.mean()) / el2nVar.std()
    localSortedIdx = torch.argsort(lossVar + el2nVar, dim=0, descending=True).cpu().numpy()
    sortedDataIdxs = np.array(dataidxs)[localSortedIdx]
    shareIdx = getIdx(sortedDataIdxs, args.numShare, exclude)
    if shareIdx is None:
        return []
    else:
        return shareIdx
