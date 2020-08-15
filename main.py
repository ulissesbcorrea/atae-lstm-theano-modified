import numpy as np
import theano
import argparse
import time
import sys
import json
import random
from Optimizer import OptimizerList
from Evaluator import Evaluators
from DataManager import DataManager
from lstm_att_con import AttentionLstm as Model

def train(model, train_data, optimizer, epoch_num, batch_size, batch_n):
    st_time = time.time()
    loss_sum = np.array([0.0, 0.0, 0.0])
    total_nodes = 0
    for batch in xrange(batch_n):
        start = batch * batch_size
        end = min((batch + 1) * batch_size, len(train_data))
        batch_loss, batch_total_nodes = do_train(model, train_data[start:end], optimizer)
        loss_sum += batch_loss
        total_nodes += batch_total_nodes

    return loss_sum[0], loss_sum[2]

def do_train(model, train_data, optimizer):
    eps0 = 1e-8
    batch_loss = np.array([0.0, 0.0, 0.0])
    total_nodes = 0
    for _, grad in model.grad.iteritems():
        grad.set_value(np.asarray(np.zeros_like(grad.get_value()), \
                dtype=theano.config.floatX))
    for item in train_data:
        sequences, target, tar_scalar, solution =  item['seqs'], item['target'], item['target_index'], item['solution']
        batch_loss += np.array(model.func_train(sequences, tar_scalar, solution))
        total_nodes += len(solution)
    for _, grad in model.grad.iteritems():
        grad.set_value(grad.get_value() / float(len(train_data)))
    optimizer.iterate(model.grad)
    return batch_loss, total_nodes

def test(model, test_data, grained):
    evaluator = Evaluators[grained]()
    keys = evaluator.keys()
    def cross(solution, pred):
        return -np.tensordot(solution, np.log(pred), axes=([0, 1], [0, 1]))

    loss = .0
    total_nodes = 0
    correct = {key: np.array([0]) for key in keys}
    wrong = {key: np.array([0]) for key in keys}
    preds_raw = []
    preds = []
    targets = []
    all_sequences = []
    solutions = []
    all_original_text = []
    for item in test_data:
        sequences, target, tar_scalar, solution, original_text =  item['seqs'], item['target'], item['target_index'], item['solution'], item['original_text']
        
        pred = model.func_test(sequences, tar_scalar)

        preds_raw.append(pred[0].tolist())
        all_sequences.append(sequences)
        targets.append(target)
        solutions.append(solution[0].tolist())
        loss += cross(solution, pred)
        total_nodes += len(solution)
        result = evaluator.accumulate(solution[-1:], pred[-1:])
        preds.append(pred[0].tolist())
        all_original_text.append(original_text)
    
    results = { 'preds_raw':preds_raw, 'preds': preds, 'sequences': all_sequences, 'targets': targets, 'text': all_original_text}
    acc = evaluator.statistic()
    return loss/total_nodes, acc, results

if __name__ == '__main__':
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='lstm')
    parser.add_argument('--seed', type=int, default=int(1000*time.time()))
    parser.add_argument('--dim_hidden', type=int, default=300)
    parser.add_argument('--dim_gram', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='data')
    parser.add_argument('--fast', type=int, choices=[0, 1], default=0)
    parser.add_argument('--screen', type=int, choices=[0, 1], default=0)
    parser.add_argument('--optimizer', type=str, default='ADAGRAD')
    parser.add_argument('--grained', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_word_vector', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=25)
    parser.add_argument('--batch', type=int, default=25)
    parser.add_argument('--patience', type=int, default=5)

    args, _ = parser.parse_known_args(argv)

    random.seed(args.seed)
    data = DataManager(args.dataset, args.seed)
    wordlist = data.gen_word()
    train_data, dev_data, test_data = data.gen_data(args.grained)
    model = Model(wordlist, argv, len(data.dict_target))
    batch_n = (len(train_data)-1) / args.batch + 1
    optimizer = OptimizerList[args.optimizer](model.params, args.lr, args.lr_word_vector)
    details = {'loss': [], 'loss_train':[], 'loss_dev':[], 'loss_test':[], \
            'acc_train':[], 'acc_dev':[], 'acc_test':[], 'loss_l2':[]}
    
    patience = args.patience
    
    best_acc_dev = 0.0
    acc_test_best_dev_model = 0.0
    best_epoch = 0
    patience_count = 0
    for e in range(args.epoch):
        random.shuffle(train_data)
        now = {}
        now['loss'], now['loss_l2'] = train(model, train_data, optimizer, e, args.batch, batch_n)
        now['loss_train'], now['acc_train'], train_results = test(model, train_data, args.grained)
        now['loss_dev'], now['acc_dev'], dev_results = test(model, dev_data, args.grained)
        now['loss_test'], now['acc_test'], test_results = test(model, test_data, args.grained)
        
        if now['acc_dev'] > best_acc_dev:
            best_acc_dev = now['acc_dev'] 
            acc_test_best_dev_model = now['acc_test'],
            best_epoch = e
            print 'New Best Acc Dev:' + str(best_acc_dev)
            print 'epoch ' + str(e)
            patience_count = 0
            all_results = {'train': train_results, 'dev': dev_results, 'test': test_results, 'epoch': e}
            with open('result/%s/%s.txt' % (seed,'best_results'), 'w') as f:
                f.writelines(json.dumps(all_results))
        else:
            patience_count = patience_count + 1
        
        if patience_count >= patience:
            print "Early stop"
            print 'epoch ' + str(e)
            
        for key, value in now.items(): 
            details[key].append(value)
        with open('result/%s.txt' % args.name, 'w') as f:
            f.writelines(json.dumps(details))
    print 'Best dev-accuracy=' + str(best_acc_dev) + ' @ epoch ' + str(best_epoch) 
    print 'test-accuracy for best dev model:' + str(best_acc_dev)