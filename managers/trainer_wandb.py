import statistics
import timeit
import os
import logging
import wandb
import pdb
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.scheduler import CosineWarmupLR
from tqdm import tqdm
from sklearn import metrics
import json
from torch.nn.utils import clip_grad_norm_


class Trainer():
    def __init__(self, params, graph_classifier, train, valid, test, train_evaluator=None, valid_evaluator=None, test_evaluator=None):
        self.params = params
        self.train_data = train
        self.valid_data = valid
        self.test_data = test
        self.graph_classifier = graph_classifier
        self.train_evaluator = train_evaluator
        self.valid_evaluator = valid_evaluator
        self.test_evaluator = test_evaluator
        self.updates_counter = 0

        ''' wandb '''
        # wandb.init(project='sumgnn-dti', entity='aida_dti', name=params.experiment_name, reinit=True)
        # wandb.config.update(params)
        # wandb.watch(self.graph_cflassifier)

        model_params = list(self.graph_classifier.parameters())
        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(model_params, lr=params.lr, weight_decay=self.params.l2)
        elif params.optimizer == "Adam":
            self.optimizer = optim.Adam(model_params, lr=params.lr, weight_decay=self.params.l2, eps = 1e-8)
        elif params.optimizer == "AdamW":
            self.optimizer = optim.AdamW(model_params, lr=params.lr, weight_decay=self.params.l2, eps = 1e-8)
        elif params.optimizer == "Momentum":
            self.optimizer = optim.SGD(model_params, lr=params.lr, momentum=0.3, weight_decay=self.params.l2)
        
        if params.dataset in ['drugbank', 'davis', 'vec']:
            self.criterion = nn.CrossEntropyLoss()
            #self.criterion = nn.BCELoss(reduce=False)
        elif params.dataset == 'BioSNAP':
            self.criterion = nn.BCELoss(reduce=False)
        
        if params.lr_scheduling:
            self.lr_scheduler = CosineWarmupLR(optimizer=self.optimizer,epochs=params.num_epochs, warmup_epochs=int(params.num_epochs*0.05),)

        self.reset_training_state()

    def reset_training_state(self):
        self.best_metric = 0
        self.last_metric = 0
        self.not_improved_count = 0

    def load_model(self):
        self.graph_classifier.load_state_dict(torch.load("my_resnet.pth"))

    def train_epoch(self):
        total_loss = 0
        all_preds = []
        all_labels = []
        all_scores = []

        dataloader = DataLoader(self.train_data, batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)
        self.graph_classifier.train()
        model_params = list(self.graph_classifier.parameters())
        bar = tqdm(enumerate(dataloader))
        for b_idx, batch in bar:
            #data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch, self.params.device)
            data_pos, r_labels_pos, targets_pos = self.params.move_batch_to_device(batch, self.params.device)
#            print(data_pos)
#            print(r_labels_pos)
#            print(targets_pos)
            self.optimizer.zero_grad()
            score_pos = self.graph_classifier(data_pos)
            if self.params.dataset == 'drugbank' :
                loss = self.criterion(score_pos, r_labels_pos)
            elif self.params.dataset == 'davis' or self.params.dataset == 'vec' :
                loss = self.criterion(score_pos, r_labels_pos)
            elif self.params.dataset == 'BioSNAP':
                m = nn.Sigmoid()
                score_pos = m(score_pos)
                targets_pos = targets_pos.unsqueeze(1)
                print(score_pos.shape)
                print(targets_pos.shape)
                loss_train = self.criterion(score_pos, r_labels_pos * targets_pos)
                loss = torch.sum(loss_train * r_labels_pos)
            loss.backward()
            clip_grad_norm_(self.graph_classifier.parameters(), max_norm=10, norm_type=2)
            self.optimizer.step()
            self.updates_counter += 1
            bar.set_description('batch: ' + str(b_idx+1) + ' / loss_train: ' + str(loss.cpu().detach().numpy()))

            # except RuntimeError:
            #     print(data_pos, r_labels_pos, targets_pos)
            #    print('-------runtime error--------')
            #    continue
            with torch.no_grad():
                total_loss += loss.item()
                if self.params.dataset != 'BioSNAP':

                    label_ids = r_labels_pos.to('cpu').numpy()
                    all_labels += label_ids.flatten().tolist()
                    #y_pred = y_pred + F.softmax(output, dim = -1)[:, -1].cpu().flatten().tolist()
                    #outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
                    all_scores += torch.argmax(score_pos, dim=1).cpu().flatten().tolist()
            if self.valid_evaluator and self.params.eval_every_iter and self.updates_counter % self.params.eval_every_iter == 0: # 
                tic = time.time()
                train_result, save_train_data = self.train_evaluator.eval()
                result, save_dev_data = self.valid_evaluator.eval()
                test_result, save_test_data = self.test_evaluator.eval()
                ''' wandb : if you don't want to use wandb, comment this part '''
                wandb.log({
                    'train_loss': train_result['loss'],
                    'train_auroc': train_result['roc_auc'],
                    'train_auprc': train_result['pr_auc'],
                    'train_acc': train_result['acc'],
                    'train_f1': train_result['f1'],
                    'val_loss': result['loss'],
                    'val_auroc': result['roc_auc'], 
                    'val_auprc': result['pr_auc'],
                    'val_acc': result['acc'],
                    'val_f1': result['f1'],
                    # 'test_loss': test_result['loss'],
                    # 'test_auroc': test_result['roc_auc'], 
                    # 'test_auprc': test_result['pr_auc'],
                    # 'test_acc': test_result['acc'],
                    # 'test_f1': test_result['f1'],
                    })
                logging.info('\033[95m Eval Performance:' + str(result) + 'in ' + str(time.time() - tic)+'\033[0m')
                logging.info('\033[93m Test Performance:' + str(test_result) + 'in ' + str(time.time() - tic)+'\033[0m')
                if result['roc_auc'] >= self.best_metric:
                    self.save_classifier()
                    self.best_metric = result['roc_auc']
                    self.not_improved_count = 0
                    if self.params.dataset != 'BioSNAP':
                        logging.info('\033[93m Test Performance Per Class:' + str(save_test_data) + 'in ' + str(time.time() - tic)+'\033[0m')
                    else:
                        with open('experiments/%s/result.json'%(self.params.experiment_name), 'a') as f:
                            f.write(json.dumps(save_test_data))
                            f.write('\n')
                else:
                    self.not_improved_count += 1
                    if self.not_improved_count > self.params.early_stop:
                        logging.info(f"Validation performance didn\'t improve for {self.params.early_stop} epochs. Training stops.")
                        break
                self.last_metric = result['roc_auc']
        if self.params.lr_scheduling:
            self.lr_scheduler.step() ### lr scheduler
        weight_norm = sum(map(lambda x: torch.norm(x), model_params))
        
        if self.params.dataset != 'BioSNAP':
            roc_auc = metrics.roc_auc_score(all_labels, all_scores, average='macro')
            precision, recall, _ = metrics.precision_recall_curve(all_labels, all_scores)
            pr_auc = metrics.auc(recall, precision)
            return total_loss/b_idx, roc_auc, pr_auc, weight_norm
        else:
            return total_loss/b_idx, 0, 0, weight_norm

    def train(self):
        self.reset_training_state()
        ''' wandb '''
        wandb.init(project='code_test', entity='aida_dti', reinit=True)

        for epoch in range(1, self.params.num_epochs + 1):
            time_start = time.time()

            loss, roc_auc, pr_auc, weight_norm = self.train_epoch()

            time_elapsed = time.time() - time_start
            logging.info(f'Epoch {epoch} with loss: {loss}, training roc_auc: {roc_auc}, training auc_pr: {pr_auc}, best validation AUC: {self.best_metric}, weight_norm: {weight_norm} in {time_elapsed}')

            # if self.valid_evaluator and epoch % self.params.eval_every == 0:
            #     result = self.valid_evaluator.eval()
            #     logging.info('\nPerformance:' + str(result))

            #     if result['roc_auc'] >= self.best_metric:
            #         self.save_classifier()
            #         self.best_metric = result['roc_auc']
            #         self.not_improved_count = 0

            #     else:
            #         self.not_improved_count += 1
            #         if self.not_improved_count > self.params.early_stop:
            #             logging.info(f"Validation performance didn\'t improve for {self.params.early_stop} epochs. Training stops.")
            #             break
            #     self.last_metric = result['roc_auc']

            if epoch % self.params.save_every == 0:
                torch.save(self.graph_classifier, os.path.join(self.params.exp_dir, 'graph_classifier_chk.pth'))

    def case_study(self):
        self.reset_training_state()
        test_result, save_test_data = self.test_evaluator.print_result()

    def save_classifier(self):
        torch.save(self.graph_classifier, os.path.join(self.params.exp_dir, 'best_graph_classifier.pth'))  # Does it overwrite or fuck with the existing file?
        logging.info('Better models found w.r.t accuracy. Saved it!')