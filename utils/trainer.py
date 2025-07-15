import os
import torch
import warnings
import pandas as pd
from models import CreateModel
from datasets import CreateDataset
from .metrics import compute_cls_metrics, compute_surv_metrics
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.cm as cm


class MetricLogger:
    def __init__(self, n_folds):
        self.fold = 0
        self.n_folds = n_folds
        self.fold_metrics = [{} for _ in range(n_folds)] # save final metrics for each fold
    
    @property
    def metrics(self):
        return list(self.fold_metrics[self.fold].keys())
    
    def _set_fold(self, fold):
        self.fold = fold
    
    def _empty_dict(self):
        return {key: 0.0 for key in self.metrics}

    def update(self, metric_dict):
        for key in metric_dict:
            self.fold_metrics[self.fold][key] = metric_dict[key]
    
    def _fold_average(self):
        if self.fold < self.n_folds - 1:
            raise Warning("Not all folds have been completed.")
        avg_metrics = self._empty_dict()
        for metric in avg_metrics:
            for fold in self.fold_metrics:
                avg_metrics[metric] += fold[metric]
            avg_metrics[metric] /= self.n_folds
        
        return avg_metrics


class Trainer:
    def __init__(self, args, wb_logger=None, val_steps=None):
        self.wb_logger = wb_logger
        self.val_steps = val_steps
        self.verbose = args.verbose
        self.m_logger = MetricLogger(n_folds=args.kfold)
    
    def _init_components(self, args):

        print(f"Train dataset size: {len(self.train_dataset)}, Test dataset size: {len(self.test_dataset)}")

        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, 
                                       shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=args.batch_size, 
                                      shuffle=False, num_workers=args.workers, drop_last=False, pin_memory=True)
        
        args.n_classes = self.train_dataset.n_classes
        args.n_features = self.train_dataset.n_features

        self.model = CreateModel(args).cuda()

        self.optimizer = getattr(torch.optim, args.optimizer)(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        if self.val_steps is None:
            self.val_steps = len(self.train_loader) 
        
        self.scheduler = None
        if args.lr_policy == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs * len(self.train_loader), eta_min=1e-6)
        elif args.lr_policy == 'cosine_restarts':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=5)
        elif args.lr_policy == 'multi_step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[18, 19], gamma=0.1)  

    def kfold_train(self, args):
        dataset = CreateDataset(args)

        for fold, (train_dataset, test_dataset) in enumerate(dataset.get_kfold_datasets()):
            self.train_dataset = train_dataset
            self.test_dataset = test_dataset
            self.fold = fold
            self.m_logger._set_fold(fold)

            self._init_components(args)

            self.train(args)

            # validate for the fold
            metric_dict = self.validate(args)
            self.m_logger.update(metric_dict)
            if self.verbose:
                print('-'*20, f'Fold {fold} Metrics', '-'*20)
            print(metric_dict)
            if args.method == 'deepcdf' and args.n_bins > 0:
                self._fold_plot_2d(args, fold, metric_dict, self.test_loader, training_set=False)
                self._fold_plot_2d(args, fold, metric_dict, self.train_loader, training_set=True)

            # self.fold_univariate_cox_regression_analysis(args, fold)
        
        avg_metrics = self.m_logger._fold_average()
        print('-'*20, 'Average Metrics', '-'*20)
        print(avg_metrics)
        self._save_fold_avg_results(args, avg_metrics)
        self.save_model(args)

    def train(self, args):
        self.model.train()
        cur_iters = 0
        for i in range(args.epochs):
            for data in self.train_loader:
                data = {k: v.cuda(non_blocking=True) if hasattr(v, 'cuda') else v for k, v in data.items()}
        
                outputs = self.model(data)
                loss = self.model.compuite_loss(outputs, data)
                print(f"Fold {self.fold} | Epoch {i} | Iter {cur_iters} | Loss: {loss.item()}")

                self.optimizer.zero_grad()
                loss.backward()

                # clip gradients to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2)

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                cur_iters += 1
                if self.verbose:
                    if cur_iters % self.val_steps == 0:

                        cur_lr = self.optimizer.param_groups[0]['lr']
                        metric_dict = self.validate(args)
                        print(f"Fold {self.fold} | Epoch {i} | Loss: {metric_dict['Loss']} | C-index: {metric_dict['C-index']} | LR: {cur_lr}")
                        if self.wb_logger is not None:
                            self.wb_logger.log({f"Fold_{self.fold}": {
                                'Train': {'loss': loss.item(), 'lr': cur_lr},
                                'Test': metric_dict
                            }})

    def validate(self, args):
        training = self.model.training
        self.model.eval()

        loss = 0.0
            
        event_indicator = torch.Tensor().cuda() # whether the event (death) has occurred
        duration = torch.Tensor().cuda()
        estimate = torch.Tensor().cuda()

        with torch.no_grad():
            for data in self.test_loader:
                data = {k: v.cuda(non_blocking=True) if hasattr(v, 'cuda') else v for k, v in data.items()}
                outputs = self.model(data)
                batch_loss = self.model.compuite_loss(outputs, data)
                loss += batch_loss.item()
                
                risk = outputs.risk
                event_indicator = torch.cat((event_indicator, data['event']), dim=0)
                duration = torch.cat((duration, data['duration']), dim=0)
                estimate = torch.cat((estimate, risk), dim=0)

            
            metric_dict = compute_surv_metrics(event_indicator, duration, estimate)
            metric_dict['Loss'] = loss / len(self.test_loader)
        
        self.model.train(training)

        return metric_dict
    
    def save_model(self, args):
        model_name = f"{args.method}_{args.backbone}.pt"
        if not os.path.exists(args.checkpoints):
            os.makedirs(args.checkpoints, exist_ok=True)
        save_path = os.path.join(args.checkpoints, model_name)
        torch.save(self.model.state_dict(), save_path)

    def _save_fold_avg_results(self, args, metric_dict, keep_best=True):
        # keep_best: whether save the best model (highest mcc) for each fold

        df_name = f"{args.kfold}Fold_{args.dataset}.xlsx"
        res_path = args.results
        if not os.path.exists(res_path):
            os.makedirs(res_path)

        settings = ['Dataset', 'Method', 'Model', 'KFold', 'Epochs', 'Seed', 'Hidden Dimensions', 'layers', 'Bins']
        kwargs = ['dataset','method', 'backbone', 'kfold', 'epochs', 'seed', 'd_hid', 'n_layers', 'n_bins']

        set2kwargs = {k: v for k, v in zip(settings, kwargs )}

        metric_names = self.m_logger.metrics
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
        
    def fold_univariate_cox_regression_analysis(self, args, fold):
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
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        
                
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
    
    def _fold_plot_2d(self, args, fold, metric_dict, dataloader, training_set, uncensored=True):
        training = self.model.training
        suffix = '_training' if training_set else ''
        self.model.eval()
        save_path = os.path.join(args.results, f"Fold_{fold}_{args.method}{suffix}.png")

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
        print(f"Fold {fold} — Biases: {biases}")

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

