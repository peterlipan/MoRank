import os
import torch
import warnings
import pandas as pd
from sksurv.util import Surv
from models import CreateModel, SurvivalQueue
from datasets import CreateDataset
from .metrics import compute_surv_metrics
from torch.utils.data import DataLoader
from .losses import RankConsistencyLoss
import numpy as np

class MetricLogger:
    def __init__(self, n_folds):
        self.n_folds = n_folds
        self.fold_metrics = [{} for _ in range(n_folds)]

    def update(self, metric_dict, fold):
        self.fold_metrics[fold] = metric_dict

    def metrics(self):
        return list(self.fold_metrics[0].keys()) if self.fold_metrics[0] else []

    def fold_average(self):
        avg_metrics = {k: 0.0 for k in self.metrics()}
        for metric in avg_metrics:
            avg_metrics[metric] = np.mean([fold[metric] for fold in self.fold_metrics])
        return avg_metrics


class Trainer:
    def __init__(self, args, wb_logger=None, val_steps=None):
        self.args = args
        self.wb_logger = wb_logger
        self.val_steps = val_steps
        self.verbose = args.verbose
        self.m_logger = MetricLogger(n_folds=args.kfold)
        os.makedirs(args.checkpoints, exist_ok=True)
        os.makedirs(args.results, exist_ok=True)
    
    def _init_components(self):
        args = self.args
        print(f"(Training) Number of samples: {self.train_dataset.n_samples}, Number of patients: {self.train_dataset.n_patients}")
        print(f"(Testing) Number of samples: {self.test_dataset.n_samples}, Number of patients: {self.test_dataset.n_patients}")

        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, collate_fn=(self.train_dataset.training_collate_fn if args.patient_level else None),
                                       shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=args.batch_size, collate_fn=(self.test_dataset.testing_collate_fn if args.patient_level else None),
                                      shuffle=False, num_workers=args.workers, drop_last=False, pin_memory=True)
        
        args.n_classes = self.train_dataset.n_classes
        args.n_features = self.train_dataset.n_features

        self.student = CreateModel(args, freeze=False, aggregator=args.aggregator).cuda()
        self.teacher = CreateModel(args, freeze=True, aggregator=args.aggregator).cuda()
        self.queue = SurvivalQueue(self.student.d_hid, args.queue_size).cuda()

        self.ranking_criteria = RankConsistencyLoss(weight=args.lambda_rank)

        self.optimizer = getattr(torch.optim, args.optimizer)(self.student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        if self.val_steps is None:
            self.val_steps = len(self.train_loader) 
        
        self.scheduler = None
        if args.lr_policy == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs * len(self.train_loader), eta_min=1e-6)
        elif args.lr_policy == 'cosine_restarts':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=5)
        elif args.lr_policy == 'multi_step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[18, 19], gamma=0.1)  
        
    def official_train_test(self):
        # non-kfold training
        args = self.args
        dataset = CreateDataset(args)
        self.train_dataset, self.test_dataset = dataset.get_official_train_test()
        self._init_components()

        self.fold = 0

        self.train()
        metric_dict = self.validate()
        self.m_logger.update(metric_dict, self.fold)
        if self.verbose:
            print('-'*20, 'Metrics', '-'*20)
        print(metric_dict)
        # self.fold_univariate_cox_regression_analysis(args, self.fold)
        self._save_fold_avg_results(metric_dict)

    def kfold_train(self):
        args = self.args
        dataset = CreateDataset(args)

        for fold, (train_dataset, test_dataset) in enumerate(dataset.get_kfold_datasets()):
            # quick test on one fold
            if hasattr(args, "fold") and fold != args.fold:
                continue
            self.train_dataset = train_dataset
            self.test_dataset = test_dataset
            self.fold = fold

            self._init_components()

            self.train()

            # validate for the fold
            metric_dict = self.validate()
            self.m_logger.update(metric_dict, fold)
            if self.verbose:
                print('-'*20, f'Fold {fold} Metrics', '-'*20)
            print(metric_dict)

            # self.fold_univariate_cox_regression_analysis(args, fold)

        avg_metrics = self.m_logger.fold_average()
        print('-'*20, 'Average Metrics', '-'*20)
        print(avg_metrics)
        self._save_fold_avg_results(avg_metrics)

    def run(self):
        args = self.args
        if args.kfold > 1:
            self.kfold_train()
        else:
            self.official_train_test()

        # save the model
        # self.save_model(args)

    def train(self):
        args = self.args
        self.student.train()
        cur_iters = 0
        for i in range(args.epochs):
            for data in self.train_loader:
                data = {k: v.cuda(non_blocking=True) if hasattr(v, 'cuda') else v for k, v in data.items()}

                stu_feat = self.student.get_features(data)
                tch_feat = self.teacher.get_features(data)
                stu_est = self.student.get_surv_stats(stu_feat)
                tch_est = self.teacher.get_surv_stats(tch_feat)
                bs = stu_feat.size(0)
                q_size = self.queue.size.item()

                # get the labels in the batch with optional patient-level aggregation
                bch_dur, bch_event, bch_label = self.teacher.aggregate_labels(data)

                # if the queue is not empty
                if args.queue and q_size > 0:
                    mem_feat, mem_event, mem_dur, mem_bins = self.queue.get()
                    # feed into the teacher model
                    mem_est = self.teacher.get_surv_stats(mem_feat)
                    # concat the risk and durations for ranking
                    logits = torch.concat([stu_est.logits, mem_est.logits], dim=0)
                    duration = torch.concat([bch_dur, mem_dur], dim=0)
                    event = torch.concat([bch_event, mem_event], dim=0)
                    label = torch.concat([bch_label, mem_bins], dim=0)
                    loss = self.student.compute_loss(logits, event, duration, label, bs)
                # otherwise, only calculate the ranking loss on the current batch
                else:
                    loss = self.student.compute_loss(stu_est.logits, bch_event, bch_dur, bch_label)

                # store the current batch into mem bank
                if args.queue:
                    self.queue.enqueue(stu_est.risk, tch_feat, bch_event, bch_dur, bch_label)

                loss += self.ranking_criteria(stu_est.logits, tch_est.logits) # ranking consistency only on the batch

                print(f"\rFold {self.fold} | Epoch {i} | Iter {cur_iters} | Loss: {loss.item()} | Q Size {q_size}", end='', flush=True)

                self.optimizer.zero_grad()
                loss.backward()

                # clip gradients to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0, norm_type=2)

                self.optimizer.step()

                # update the teacher model
                self.teacher.ema_update(self.student, cur_iters)

                if self.scheduler is not None:
                    self.scheduler.step()

                cur_iters += 1
                if self.verbose:
                    if cur_iters % self.val_steps == 0:

                        cur_lr = self.optimizer.param_groups[0]['lr']
                        metric_dict = self.validate()
                        print('\n', '-'*20, 'Metrics', '-'*20)
                        for key, value in metric_dict.items():
                            print(f"{key}: {value}")
                        if self.wb_logger is not None:
                            self.wb_logger.log({f"Fold_{self.fold}": {
                                'Train': {'loss': loss.item(), 'lr': cur_lr},
                                'Test': metric_dict
                            }})

    def validate(self):  # patient-level evaluation
        args = self.args
        training = self.student.training
        self.student.eval()

        loss = 0.0

        event_indicator = []
        duration = []
        risk_prob = []
        surv_prob = []
        patient_idx = []

        # for DeepSurv baseline survival
        if args.method.lower() == 'deepsurv':
            bin_times = torch.arange(self.train_dataset.n_classes, dtype=torch.float32)
            self.student.prepare_for_validation(
                self.train_loader, bin_times.cuda()
            )

        # training survival data for IPCW
        train_duration = self.train_dataset.duration
        train_event = self.train_dataset.event.astype(bool)
        train_surv = Surv.from_arrays(event=train_event, time=train_duration)

        with torch.no_grad():
            for data in self.test_loader:
                data = {k: v.cuda(non_blocking=True) if hasattr(v, 'cuda') else v for k, v in data.items()}
                outputs = self.student(data)
                # get the labels in the batch with optional patient-level aggregation
                bch_dur, bch_event, bch_label = self.student.aggregate_labels(data)
                batch_loss = self.student.compute_loss(outputs.logits, bch_event, bch_dur, bch_label)
                loss += batch_loss.item()

                risk = outputs.risk

                event_indicator.append(data['event'].cpu().numpy())
                duration.append(data['duration'].cpu().numpy())
                risk_prob.append(risk.cpu().numpy())
                surv_prob.append(outputs.surv.cpu().numpy())
                patient_idx.extend(data['patient_id'])

        event_indicator = np.concatenate(event_indicator).astype(bool)
        duration = np.concatenate(duration)
        risk_prob = np.concatenate(risk_prob)
        surv_prob = np.concatenate(surv_prob)
        patient_idx = np.array(patient_idx)

        if args.metric_agg != 'None':
            agg = args.metric_agg.lower()

            patient_risk, patient_surv, patient_event, patient_time = [], [], [], []
            for pid in np.unique(patient_idx):
                mask = patient_idx == pid
                risks = risk_prob[mask]
                survs = surv_prob[mask]

                if agg == "mean":
                    patient_risk.append(np.mean(risks))
                    patient_surv.append(np.mean(survs, axis=0))
                elif agg == "max": # take the sample with the highest risk as the representative one
                    i = np.argmax(risks)
                    patient_risk.append(risks[i])
                    patient_surv.append(survs[i])
                elif agg == "min": # take the sample with the lowest risk as the representative one
                    i = np.argmin(risks)
                    patient_risk.append(risks[i])
                    patient_surv.append(survs[i])

                # assume event/time consistent per patient
                patient_event.append(event_indicator[mask][0])
                patient_time.append(duration[mask][0])

            risk_prob = np.array(patient_risk)
            event_indicator = np.array(patient_event)
            duration = np.array(patient_time)
            surv_prob = np.array(patient_surv)

        test_surv = Surv.from_arrays(event=event_indicator, time=duration)
        n_eval = int(args.n_bins * 10) if args.n_bins > 0 else 3000
        # Earliest event
        min_time = duration[event_indicator].min()

        # Last evaluable time = largest event time strictly less than max censoring
        censor_mask = ~event_indicator
        if censor_mask.any():
            max_censor_time = duration[censor_mask].max()
            max_time = duration[(event_indicator) & (duration < max_censor_time)].max()
        else:
            max_time = duration.max()  # no censoring case

        # Sample times within [min_time, max_time]
        time_points = np.linspace(min_time, max_time, n_eval)
        time_labels = self.test_dataset._duration_to_label(time_points)

        surv_prob = surv_prob[:, time_labels]

        metric_dict = compute_surv_metrics(train_surv, test_surv, risk_prob, surv_prob, time_points)
        metric_dict['Loss'] = loss / len(self.test_loader)

        self.student.train(training)
        return metric_dict

    def save_model(self):
        args = self.args
        model_name = f"{args.method}_{args.backbone}.pt"
        save_path = os.path.join(args.checkpoints, model_name)
        torch.save(self.student.state_dict(), save_path)

    def _save_fold_avg_results(self, metric_dict, keep_best=True):
        # keep_best: whether save the best model (highest mcc) for each fold
        args = self.args
        suffix = '_image-level' if args.aggregator == 'None' and args.metric_agg == 'None' else '_patient-level'
        df_name = f"{args.kfold}Fold_{args.dataset}{suffix}.xlsx"
        res_path = args.results

        settings = ['Dataset', 'Method', 'Model', 'KFold', 'Epochs', 'Seed', 'Bins', 'Evaluation Aggregation', 'Model Aggregator']
        kwargs = ['dataset','method', 'backbone', 'kfold', 'epochs', 'seed', 'n_bins', 'metric_agg', 'aggregator']

        set2kwargs = {k: v for k, v in zip(settings, kwargs )}

        metric_names = self.m_logger.metrics()
        df_columns = settings + metric_names
        
        df_path = os.path.join(res_path, df_name)
        if not os.path.exists(df_path):
            df = pd.DataFrame(columns=df_columns)
        else:
            df = pd.read_excel(df_path)
            if df_columns != df.columns.tolist():
                warnings.warn("Columns in the existing excel file do not match the current settings.")
                df = pd.DataFrame(columns=df_columns)
        
        new_row = {k: args.__dict__[v] for k, v in set2kwargs.items()}

        if keep_best: # keep the rows with the best mcc for each fold
            reference = 'C-index'
            exsiting_rows = df[(df[settings] == pd.Series(new_row)).all(axis=1)]
            if not exsiting_rows.empty:
                exsiting_mcc = exsiting_rows[reference].values
                if metric_dict[reference] > exsiting_mcc:
                    df = df.drop(exsiting_rows.index)
                else:
                    return

        new_row.update(metric_dict)
        df = df._append(new_row, ignore_index=True)
        df.to_excel(df_path, index=False)

    def fold_univariate_cox_regression_analysis(self):
        args = self.args
        fold = self.fold
        training = self.model.training
        self.model.eval()

        event_indicator = torch.empty(0).cuda()
        duration = torch.empty(0).cuda()
        risk_factor = torch.empty(0).cuda()
        filename = []
        patient_id = []

        df_name = f"{args.kfold}Fold_{args.dataset}_Cox.xlsx"
        res_path = args.results
        df_path = os.path.join(res_path, df_name)
        
                
        with torch.no_grad():
            for data in self.test_loader:
                data = {k: v.cuda(non_blocking=True) if hasattr(v, 'cuda') else v for k, v in data.items()}
                outputs = self.model(data)
                risk = outputs.risk
                event_indicator = torch.cat((event_indicator, data['event']), dim=0)
                duration = torch.cat((duration, data['duration']), dim=0)
                risk_factor = torch.cat((risk_factor, risk), dim=0)
                filename.extend(data['filename'])
                patient_id.extend(data['patient_id'])
        
        event_indicator = event_indicator.cpu().numpy()
        duration = duration.cpu().numpy()
        risk_factor = risk_factor.cpu().numpy()
                

        fold_df = pd.DataFrame({
            'BBNumber': patient_id,
            'Filename': filename,
            'Fold': [fold] * len(filename),
            'event': event_indicator,
            'duration': duration,
            f'{args.include}_{args.backbone}_{args.surv_loss}': risk_factor,
        })

        if hasattr(self, 'cox_df'):
            self.cox_df = pd.concat([self.cox_df, fold_df], ignore_index=True)
        else:
            self.cox_df = fold_df

        if fold == args.kfold - 1:
            if os.path.exists(df_path):
                existing_df = pd.read_excel(df_path)
                existing_df[f'{args.backbone}'] = None  # Initialize the new column

                for _, row in self.cox_df.iterrows():
                    filename = row['Filename']
                    if filename in existing_df['Filename'].values:
                        existing_df.loc[existing_df['Filename'] == filename, f'{args.include}_{args.backbone}_{args.surv_loss}'] = row[f'{args.include}_{args.backbone}_{args.surv_loss}']
            else:
                existing_df = self.cox_df
            existing_df.to_excel(df_path, index=False)

        self.model.train(training)
