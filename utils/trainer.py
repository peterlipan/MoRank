import os
import torch
import warnings
import pandas as pd
from sksurv.util import Surv
from models import CreateModel, SurvivalQueue
from datasets import CreateDataset
from .metrics import ordinal_metrics, compute_surv_metrics
from torch.utils.data import DataLoader
from .losses import RankConsistencyLoss
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.cm as cm


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
        if args.method == 'deepcdf' and args.n_bins > 0:
            self._fold_plot_2d(metric_dict, self.test_loader, training_set=False)
            self._fold_plot_2d(metric_dict, self.train_loader, training_set=True)
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
            if args.method == 'deepcdf' and args.n_bins > 0:
                self._fold_plot_2d(metric_dict, self.test_loader, training_set=False)
                self._fold_plot_2d(metric_dict, self.train_loader, training_set=True)
                self.plot_risk_cdf_curves(self.train_loader, training_set=True)
                self.plot_risk_cdf_curves(self.test_loader, training_set=False)

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

    def cls_validate(self):
        args = self.args
        training = self.student.training

        loss = 0.0

        ground_truth = torch.Tensor().cuda()
        predictions = torch.Tensor().cuda()

        with torch.no_grad():
            self.student.eval()
            for data in self.test_loader:
                data = {k: v.cuda(non_blocking=True) if hasattr(v, 'cuda') else v for k, v in data.items()}
                outputs = self.student(data)
                batch_loss = self.student.compute_loss(outputs, data)
                loss += batch_loss.item()

                ground_truth = torch.cat((ground_truth, data['label']), dim=0)
                predictions = torch.cat((predictions, outputs.y_pred), dim=0)
            alpha = getattr(args, 'alpha', 0.5)
            metric_dict = ordinal_metrics(ground_truth, predictions, alpha)
            metric_dict['Loss'] = loss / len(self.test_loader)
        self.student.train(training)
        return metric_dict

    def surv_validate(self):  # patient-level evaluation
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

        if hasattr(args, "aggregate"):
            agg = args.aggregate.lower()

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
                else:
                    raise ValueError(f"Unknown aggregation method: {agg}")

                # assume event/time consistent per patient
                patient_event.append(event_indicator[mask][0])
                patient_time.append(duration[mask][0])

            risk_prob = np.array(patient_risk)
            event_indicator = np.array(patient_event)
            duration = np.array(patient_time)
            surv_prob = np.array(patient_surv)

        test_surv = Surv.from_arrays(event=event_indicator, time=duration)
        n_eval = int(args.n_bins * 10) if args.n_bins > 0 else 3000
        valid_durations = duration[event_indicator]
        time_points = np.linspace(valid_durations.min(), valid_durations.max() - 1, n_eval)
        time_labels = self.test_dataset._duration_to_label(time_points)

        surv_prob = surv_prob[:, time_labels] if surv_prob is not None else None

        metric_dict = compute_surv_metrics(train_surv, test_surv, risk_prob, surv_prob, time_points)
        metric_dict['Loss'] = loss / len(self.test_loader)

        self.student.train(training)
        return metric_dict

    def validate(self):
        args = self.args
        if args.task.lower() == 'classification':
            metric_dict = self.cls_validate()
        elif args.task.lower() == 'survival':
            metric_dict = self.surv_validate()
        else:
            raise ValueError(f"Unknown task: {args.task}. Supported tasks are: classification, survival.")

        return metric_dict

    def save_model(self):
        args = self.args
        model_name = f"{args.method}_{args.backbone}.pt"
        save_path = os.path.join(args.checkpoints, model_name)
        torch.save(self.student.state_dict(), save_path)

    def _save_fold_surv_avg_results(self, metric_dict, keep_best=True):
        # keep_best: whether save the best model (highest mcc) for each fold
        args = self.args
        df_name = f"{args.kfold}Fold_{args.dataset}.xlsx"
        res_path = args.results

        settings = ['Dataset', 'Method', 'Model', 'KFold', 'Epochs', 'Seed', 'Bins']
        kwargs = ['dataset','method', 'backbone', 'kfold', 'epochs', 'seed', 'n_bins']

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
        
    def _save_fold_cls_avg_results(self, metric_dict, keep_best=True):
        # keep_best: whether save the best model (highest mcc) for each fold
        args = self.args
        df_name = f"{args.kfold}Fold_{args.dataset}_Classification.xlsx"
        res_path = args.results

        settings = ['Dataset', 'Method', 'Model', 'KFold', 'Epochs', 'Seed']
        kwargs = ['dataset','method', 'backbone', 'kfold', 'epochs', 'seed']

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

        if keep_best:
            reference = 'Acc'
            existing_rows = df[(df[settings] == pd.Series(new_row)).all(axis=1)]
            if not existing_rows.empty:
                existing_acc = existing_rows[reference].values
                if metric_dict[reference] > existing_acc:
                    df = df.drop(existing_rows.index)
                else:
                    return

        new_row.update(metric_dict)
        df = df._append(new_row, ignore_index=True)
        df.to_excel(df_path, index=False)

    def _save_fold_avg_results(self, metric_dict):
        # save the average results for each fold
        args = self.args
        if args.task.lower() == 'classification':
            self._save_fold_cls_avg_results(metric_dict)
        elif args.task.lower() == 'survival':
            self._save_fold_surv_avg_results(metric_dict)
        else:
            raise ValueError(f"Unknown task: {args.task}. Supported tasks are: classification, survival.")

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
    
    def _fold_plot_2d(self, metric_dict, dataloader, training_set, uncensored=True):
        args = self.args
        fold = self.fold
        training = self.model.training
        suffix = '_training' if training_set else ''
        self.model.eval()
        save_path = os.path.join(args.results, args.dataset, f"Fold_{fold}_{args.method}{suffix}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Storage
        coord_x = torch.Tensor().cuda()
        coord_y = torch.Tensor().cuda()
        event = torch.Tensor().cuda()
        duration = torch.Tensor().cuda()

        # Collect data
        with torch.no_grad():
            for data in dataloader:
                data = {k: v.cuda(non_blocking=True) if hasattr(v, 'cuda') else v for k, v in data.items()}
                outputs = self.model.project_2d(data)
                coord_x = torch.cat((coord_x, outputs.x), dim=0)
                coord_y = torch.cat((coord_y, outputs.y), dim=0)
                event = torch.cat((event, data['event']), dim=0)
                duration = torch.cat((duration, data['duration']), dim=0)

        # Convert to NumPy
        coord_x = coord_x.cpu().numpy()
        coord_y = coord_y.cpu().numpy()
        event = event.cpu().numpy()
        duration = duration.cpu().numpy()
        biases = self.model.biases.cpu().detach().numpy()
        # print(f"Fold {fold} — Biases: {biases}")

        # Normalize duration for red colormap
        norm = Normalize(vmin=duration[event == 1].min(), vmax=duration[event == 1].max())
        cmap = cm.Reds

        # Prepare figure
        plt.figure(figsize=(8, 6))

        # Plot censored (event=0) in blue
        if not uncensored:
            plt.scatter(coord_x[event == 0], coord_y[event == 0], color='blue', alpha=0.5, label='Censored')

        # Plot events (event=1) in red with colormap based on duration
        colors = cmap(norm(duration[event == 1]))
        plt.scatter(coord_x[event == 1], coord_y[event == 1], color=colors, alpha=0.7, label='Event')

        # Add colorbar for duration
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('Survival Time')

        # Plot vertical threshold lines (CDF biases)
        for b in biases:
            plt.axvline(x=-b, color='black', linestyle='--', alpha=0.5)

        # Plot formatting
        plt.xlabel("Projection (along head weight)")
        plt.ylabel("Perpendicular Component")
        plt.title(f"Fold {fold} — C-Index: {metric_dict['C-index']:.4f}")
        plt.legend(loc='upper right')
        plt.grid(False)

        # Save and cleanup
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Restore training state
        self.model.train(training)

    def plot_risk_cdf_curves(self, dataloader, training_set=False):
        args = self.args
        fold = self.fold
        self.model.eval()
        training = self.model.training
        suffix = '_training' if training_set else ''
        save_path = os.path.join(args.results, args.dataset, f"Fold_{fold}_CDFRiskCurves{suffix}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        cdfs, labels = [], []
        with torch.no_grad():
            for data in dataloader:
                data = {k: v.cuda(non_blocking=True) if hasattr(v, 'cuda') else v for k, v in data.items()}
                outputs = self.model(data)  # Should return logits internally passed through sigmoid → CDF
                cdfs.append(outputs.cdf.cpu())
                labels.append(data['label'].cpu())

        # Stack everything
        cdfs = torch.cat(cdfs, dim=0).numpy()
        labels = torch.cat(labels, dim=0).numpy()

        # Sort by risk
        lowest_time_idx = np.min(labels)
        highest_time_idx = np.max(labels)
        median_time_idx = np.median(labels).astype(int)

        lowest_avg_cdf = np.mean(cdfs[labels == lowest_time_idx], axis=0)
        highest_avg_cdf = np.mean(cdfs[labels == highest_time_idx], axis=0)
        median_avg_cdf = np.mean(cdfs[labels == median_time_idx], axis=0)

        # Time bins
        time_bins = np.arange(cdfs.shape[1])

        # Plot
        plt.figure(figsize=(8, 6))

        for cdf, color, time_idx in zip([lowest_avg_cdf, median_avg_cdf, highest_avg_cdf],
                                    ['red', 'orange', 'green'],
                                    [lowest_time_idx, median_time_idx, highest_time_idx]):
            plt.plot(time_bins, cdf, label=f"Label = {time_idx}", color=color)
            # plt.axvline(labels[idx], color=color, linestyle='--', alpha=0.7)

        plt.xlabel("Time Bin Index")
        plt.ylabel("Predicted CDF")
        plt.title(f"Fold {fold} — CDF Curves by Risk Stratification")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.model.train(training)
