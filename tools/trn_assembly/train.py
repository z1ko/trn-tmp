import os
import os.path as osp
import sys
import time
import argparse
import einops
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import torch.utils.benchmark as benchmark

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path('/home/fziche/develop/TRN.pytorch/lib')

import utils as utl

from utils.multicrossentropy_loss import CEplusMSE
from models import build_model
from datasets.assembly101_data_layer import Assembly101Dataset

INPUT='poses' # poses - embeddings

ACTION_TYPE='noun'
ACTION_CATEGORIES = {
    'verb': ('verb', 25),
    'noun': ('noun', 91)
}

class CEplusMSE(nn.Module):
    """
    Loss from MS-TCN paper. CrossEntropy + MSE
    https://arxiv.org/abs/1903.01945
    """

    def __init__(self, num_classes, weight, alpha=0.17):
        super().__init__()

        self.ce = nn.CrossEntropyLoss(ignore_index=-100, weight=weight)
        self.mse = nn.MSELoss(reduction='none')
        self.classes = num_classes
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        :param logits:  [batch_size, seq_len, logits]
        :param targets: [batch_size, seq_len]
        """

        logits = einops.rearrange(logits, 'batch_size seq_len logits -> batch_size logits seq_len')
        loss = { }

        # Frame level classification
        loss['loss_ce'] = self.ce(
            einops.rearrange(logits, "batch_size logits seq_len -> (batch_size seq_len) logits"),
            einops.rearrange(targets, "batch_size seq_len -> (batch_size seq_len)")
        )

        # Neighbour frames should have similar values
        loss['loss_mse'] = torch.mean(torch.clamp(self.mse(
            F.log_softmax(logits[:, :, 1:], dim=1),
            F.log_softmax(logits.detach()[:, :, :-1], dim=1)
        ), min=0.0, max=160.0))

        loss['loss_total'] = loss['loss_ce'] + self.alpha * loss['loss_mse']
        return loss

def calculate_multi_loss(logits, targets, categories, alpha, weights):
    """ Calculate ce+mse multiloss between different target categories
            logits:     an array of tensors, each of the shape [batch_size, seq_len, logits]
            targets:    an array of tensors, each of the shape [batch_size, seq_len]
            categories: [('verb', 25), ('noun', 90)]
    """

    result = { }
    for category_name, _ in categories:
        result[category_name] = {}

    combined = 0.0
    for i, (logit, target, category) in enumerate(zip(logits, targets, categories)):
#        assert(target.shape[0] == 1)
        category_name, num_classes = category
        loss = CEplusMSE(num_classes, alpha=alpha, weight=weights[i+1]) # weights[i+1]
        category_result = loss(logit, target)
        result[category_name] = category_result
        
        # Accumulated loss between all categories
        combined += category_result['loss_total']

    return result, combined

def _get_labels_start_end_time(frame_wise_labels, ignored_classes=[-100]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in ignored_classes:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in ignored_classes:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in ignored_classes:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in ignored_classes:
        ends.append(i + 1)
    return labels, starts, ends

class MeanOverFramesAccuracy:
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        :param predictions: [batch_size, seq_len]
        :param targets: [batch_size, seq_len]
        """

        predictions, targets = np.array(predictions.cpu()), np.array(targets.cpu())

        # Skip all padding
        mask = np.logical_not(np.isin(targets, [-100]))

        total = mask.sum()
        correct = (predictions == targets)[mask].sum()
        result = correct / total if total != 0 else 0
        return result


class F1Score:
    def __init__(self, overlaps = [0.1, 0.25, 0.5]):
        self.overlaps = overlaps

    def __call__(self, predictions, targets) -> float:
        """
        :param predictions: [batch_size, seq_len]
        :param targets: [batch_size, seq_len]
        """

        #self.tps = np.zeros((len(self.overlaps), self.classes))
        #self.fps = np.zeros((len(self.overlaps), self.classes))
        #self.fns = np.zeros((len(self.overlaps), self.classes))

        result = {}
        for o in self.overlaps:
            result[f'F1@{int(o*100)}'] = 0.0

        batches_count = predictions.shape[0]
        predictions, targets = np.array(predictions.cpu()), np.array(targets.cpu())
        for p, t in zip(predictions, targets):

            # Skip all padding
            mask = np.logical_not(np.isin(t, [-100]))
            t = t[mask]
            p = p[mask]

            for i, overlap in enumerate(self.overlaps):
                tp, fp, fn = self.f_score(
                    p.tolist(),
                    t.tolist(),
                    overlap
                )
                
                #self.tps[i] += tp
                #self.fps[i] += fp
                #self.fns[i] += fn

                f1 = self.get_f1_score(tp, fp, fn)
                result[f'F1@{int(overlap*100)}'] += f1

        for o in self.overlaps:
            result[f'F1@{int(o*100)}'] /= batches_count 
        return result

    @staticmethod
    def f_score(predictions, targets, overlap, ignore_classes=[-100]):
        p_label, p_start, p_end = _get_labels_start_end_time(predictions, ignore_classes)
        y_label, y_start, y_end = _get_labels_start_end_time(targets, ignore_classes)

        tp = 0
        fp = 0

        hits = np.zeros(len(y_label))

        for j in range(len(p_label)):
            intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
            union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
            IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
            # Get the best scoring segment
            idx = np.array(IoU).argmax()

            if IoU[idx] >= overlap and not hits[idx]:
                tp += 1
                hits[idx] = 1
            else:
                fp += 1
        fn = len(y_label) - sum(hits)
        return float(tp), float(fp), float(fn)

    @staticmethod
    def get_f1_score(tp, fp, fn):
        if tp + fp != 0.0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
        else:
            precision = 0.0
            recall = 0.0
        
        if precision + recall != 0.0:
            return 2.0 * (precision * recall) / (precision + recall)
        else:
            return 0.0

class EditDistance:
    def __init__(self, normalize):
        self.normalize = normalize

    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        :param predictions: [batch_size, seq_len]
        :param targets: [batch_size, seq_len]
        """

        batch_scores = []
        predictions, targets = np.array(predictions.cpu()), np.array(targets.cpu())
        for pred, target in zip(predictions, targets):

            # Skip all padding
            mask = np.logical_not(np.isin(target, [-100]))
            target = target[mask]
            pred = pred[mask]

            batch_scores.append(self.edit_score(
                predictions=pred.tolist(),
                targets=target.tolist(),
                norm=self.normalize
            ))

        # Mean in the batch
        return sum(batch_scores) / len(batch_scores) * 0.01
    
    @staticmethod
    def edit_score(predictions, targets, norm=True, ignore_classes=[-100]):
        P, _, _ = _get_labels_start_end_time(predictions, ignore_classes)
        Y, _, _ = _get_labels_start_end_time(targets, ignore_classes)
        return EditDistance.levenstein(P, Y, norm)
    
    @staticmethod
    def levenstein(p, y, norm=False):
        m_row = len(p) 
        n_col = len(y)
        D = np.zeros([m_row+1, n_col+1], float)
        for i in range(m_row+1):
            D[i, 0] = i
        for i in range(n_col+1):
            D[0, i] = i

        for j in range(1, n_col+1):
            for i in range(1, m_row+1):
                if y[j-1] == p[i-1]:
                    D[i, j] = D[i-1, j-1]
                else:
                    D[i, j] = min(D[i-1, j] + 1,
                                D[i, j-1] + 1,
                                D[i-1, j-1] + 1)
        
        if norm:
            score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
        else:
            score = D[-1, -1]

        return score

def calculate_multi_metrics(logits, targets, categories):
    """ Calculate ce+mse multiloss between different target categories
    """

    result = {}
    for category_name, _ in categories:
        result[category_name] = {}

    for logit, target, category in zip(logits, targets, categories):
        category_name, _ = category
        category_result = calculate_metrics(logit, target)
        result[category_name] = category_result
        
    return result

def calculate_metrics(logits, targets, config=None):
    """ logits:  [batch_size, seq_len, logits]
        targets: [batch_size, seq_len]
    """

    mof = MeanOverFramesAccuracy()
    f1 = F1Score()
    edit = EditDistance(True)

    probabilities = torch.softmax(logits, dim=-1)
    predictions = torch.argmax(probabilities, dim=-1)

    result = { 'mof': mof(predictions, targets), 'edit': edit(predictions, targets) }
    result.update(f1(predictions, targets))
    return result

def main(args):

    args.modality = INPUT

    #this_dir = osp.join(osp.dirname(__file__), '.')
    #save_dir = osp.join(this_dir, 'checkpoints')
    #if not osp.isdir(save_dir):
    #    os.makedirs(save_dir)

    #command = 'python ' + ' '.join(sys.argv)
    #logger = utl.setup_logger(osp.join(this_dir, 'log.txt'), command=command)
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utl.set_seed(420)


    model = build_model(args)

    #if osp.isfile(args.checkpoint):
    #    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    #    model.load_state_dict(checkpoint['model_state_dict'])
    #else:

    model.apply(utl.weights_init)
    
    #if args.distributed:
    #    model = nn.DataParallel(model)
    
    torch.cuda.reset_max_memory_allocated()
    model = model.to(device)
    memory = torch.cuda.max_memory_allocated()
    print('model_size_allocated: ', memory / 1e6, '(MB)')

    params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total parameters: ', params_count)

    optimizer = optim.Adam(model.parameters(), lr=5e-04, weight_decay=5e-04)
    #if osp.isfile(args.checkpoint):
    #    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #    for param_group in optimizer.param_groups:
    #        param_group['lr'] = args.lr
    #    args.start_epoch += checkpoint['epoch']
    #softmax = nn.Softmax(dim=1).to(device)

    # weights of labels
    with open('data/Assembly101/fine-labels-weights.pkl','rb') as f:
        weights = [torch.tensor(x, dtype=torch.float32).to('cuda') for x in pickle.load(f)]

    CATEGORIES = [ACTION_CATEGORIES[ACTION_TYPE]]

    for epoch in range(21+1):
        if epoch == 21:
            args.lr = args.lr * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

        #def collate(data):
        #    embeddings, poses, enc_labels, dec_labels = [], [], []
        #    for e, p, el, dl in data:
        #        embeddings.append(e)
        #        poses.append(p)
        #        enc_labels.append(el)
        #        dec_labels.append(dl)
        #    embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
        #    poses = torch.nn.utils.rnn.pad_sequence(poses, batch_first=True)
        #    enc_labels = torch.nn.utils.rnn.pad_sequence(enc_labels, batch_first=True, padding_value=-100)
        #    dec_labels = torch.nn.utils.rnn.pad_sequence(dec_labels, batch_first=True, padding_value=-100)
        #    return embeddings, poses, labels[..., 1:] # take only verb + noun

        data_loaders = {
            'validation': torch.utils.data.DataLoader(
                dataset=Assembly101Dataset(mode='validation', config=None, target=ACTION_TYPE),
                batch_size=1,
                shuffle=False,
                #collate_fn=collate,
                num_workers=16,
            ),
            'train': torch.utils.data.DataLoader(
                dataset=Assembly101Dataset(mode='train', config=None, target=ACTION_TYPE),
                batch_size=16,
                shuffle=True,
                #collate_fn=collate,
                num_workers=16,
            )
        }

        start = time.time()
        for phase in ['train', 'validation']:
            training = phase=='train'

            if training:
                #TODO: REMOVE
                continue

                model.train(True)
                with torch.set_grad_enabled(training):
                    for batch_idx, (embeddings, poses, enc_target, dec_target) \
                            in enumerate(tqdm(data_loaders[phase]), start=0):
                        
                        batch_size = embeddings.shape[0]

                        if INPUT == 'embeddings':
                            x = embeddings.to(device)
                        else:
                            x = poses.to(device)
                            x = einops.rearrange(x, 'B T J F -> B T (J F)')

                        enc_target = enc_target.to(device).long()
                        dec_target = dec_target.to(device).long()

                        enc_score, dec_score = model(x)
                        enc_score = enc_score[None, ...]
                        dec_score = dec_score[None, ...]

                        # Frame encoder loss
                        _, enc_loss = calculate_multi_loss([enc_score], [enc_target], 
                                                        categories=CATEGORIES, 
                                                        alpha=0.22, 
                                                        weights=weights)

                        # Future frame decoder loss
                        dec_target = einops.rearrange(dec_target, 'B S Y -> B (S Y)')
                        _, dec_loss = calculate_multi_loss([dec_score], [dec_target], 
                                                        categories=CATEGORIES, 
                                                        alpha=0.22, 
                                                        weights=weights)
                        
                        optimizer.zero_grad()
                        loss = enc_loss + dec_loss
                        loss.backward()
                        optimizer.step()
                        
            else:

                result = {
                    'mof': 0.0,
                    'edit': 0.0,
                    'F1@10': 0.0,
                    'F1@25': 0.0,
                    'F1@50': 0.0
                }

                model.train(False)
                with torch.set_grad_enabled(False):
                    for batch_idx, (embeddings, poses, enc_target, dec_target) \
                        in enumerate(tqdm(data_loaders[phase]), start=0):
                        assert(enc_target.shape[0] == 1)

                        if INPUT == 'embeddings':
                            x = embeddings.to(device)
                        else:
                            x = poses.to(device)
                            x = einops.rearrange(x, 'B T J F -> B T (J F)')

                        # State
                        enc_hx = torch.zeros((1,model.hidden_size)).to('cuda')
                        enc_cx = torch.zeros((1,model.hidden_size)).to('cuda')
                        future_input = torch.zeros((1,model.future_size)).to('cuda')

                        # Benchmark
                        if True:
                            current_input = x[:, 0, :]
                            t0 = benchmark.Timer(
                                stmt='model.step(current_input, None, future_input, enc_hx, enc_cx)',
                                globals={
                                    'model': model,
                                    'current_input': current_input,
                                    'future_input': future_input,
                                    'enc_hx': enc_hx,
                                    'enc_cx': enc_cx
                                }
                            )
                         
                            print("state_size: ", (enc_hx.numel() + enc_cx.numel() + future_input.numel()) * 4 / 1e3, "(KB)")
                            print("elapsed_time: ", t0.timeit(100), "(ms)")

                        predictions = []
                        for l in range(embeddings.shape[1]):
                            current_input = x[:, l, :]
                            future_input, enc_hx, enc_cx, enc_score, dec_score_stack = \
                                model.step(current_input, None, future_input, enc_hx, enc_cx)
                            
                            enc_prob = torch.softmax(enc_score, dim=-1)
                            predictions.append(enc_prob[0])

                        predictions = torch.stack(predictions)
                        predictions = predictions[None, ...]

                        metrics = calculate_multi_metrics([predictions], [enc_target], CATEGORIES)
                        result = { k: result[k] + v for k, v in metrics[ACTION_TYPE].items() }
                    
                    elements_count = len(data_loaders[phase])
                    result = { k: v / elements_count for k, v in result.items() }
                    print(f'action type: {ACTION_TYPE} -> {result}')


        end = time.time()

        #if args.debug:
        #    result_file = 'inputs-{}-epoch-{}.json'.format(args.inputs, epoch)
        #    # Compute result for encoder
        #    enc_mAP = utl.compute_result_multilabel(
        #        args.class_index,
        #        enc_score_metrics,
        #        enc_target_metrics,
        #        save_dir,
        #        result_file,
        #        ignore_class=[0,21],
        #        save=True,
        #    )
        #    # Compute result for decoder
        #    dec_mAP = utl.compute_result_multilabel(
        #        args.class_index,
        #        dec_score_metrics,
        #        dec_target_metrics,
        #        save_dir,
        #        result_file,
        #        ignore_class=[0,21],
        #        save=False,
        #    )

        # Output result
        #logger.output(epoch, enc_losses, dec_losses,
        #              len(data_loaders['train'].dataset), len(data_loaders['test'].dataset),
        #              enc_mAP, dec_mAP, end - start, debug=args.debug)

        # Save model
        #checkpoint_file = 'inputs-{}-epoch-{}.pth'.format(args.inputs, epoch)
        #torch.save({
        #    'epoch': epoch,
        #    'model_state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #}, osp.join(save_dir, checkpoint_file))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--modality', default='poses', type=str)

    parser.add_argument('--num_classes', default=ACTION_CATEGORIES[ACTION_TYPE][1], type=int)

    parser.add_argument('--data_info', default='data/data_info.json', type=str)
    parser.add_argument('--checkpoint', default='', type=str)
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--epochs', default=21, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', default=5e-04, type=float)
    parser.add_argument('--weight_decay', default=5e-04, type=float)
    parser.add_argument('--seed', default=25, type=int)
    parser.add_argument('--phases', default=['train', 'test'], type=list)

    parser.add_argument('--dataset', default='Assembly', type=str)
    parser.add_argument('--data_root', default='data/THUMOS', type=str)
    parser.add_argument('--model', default='TRN', type=str)
    parser.add_argument('--inputs', default='multistream', type=str)
    parser.add_argument('--hidden_size', default=256, type=int) # was 1024
    parser.add_argument('--camera_feature', default='resnet200-fc', type=str)
    parser.add_argument('--motion_feature', default='bn_inception', type=str)
    parser.add_argument('--enc_steps', default=64, type=int)
    parser.add_argument('--dec_steps', default=8, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)

    main(parser.parse_args())
