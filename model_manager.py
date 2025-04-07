from collections import deque
import torch.multiprocessing as mp
import torch
import threading
from typing import List, Tuple
import numpy as np
from model import CompleteModel
from feature import SubQueryEncoder, QueryFeature
from utils import get_table_aliases
from model import CombinedLoss
import time
import traceback
from config import config
import os
device = "cuda" if torch.cuda.is_available() else "cpu"

class ModelManager:
    def __init__(self, model_dim, query_encoder: SubQueryEncoder):
        # Set multiprocessing start method
        mp.set_start_method('spawn', force=True)
        
        # Initialize two models
        if config.model_save_dir:
            self.model_save_path = config.model_save_dir + f'loss{config.delta_weight}_layer{config.num_layers}_learnrate{config.learn_rate}_{config.num_epochs_per_train}per_train.pth'
        else:
            self.model_save_path = None

        print(f"num_layers: {config.num_layers}")
        print(f"learn_rate: {config.learn_rate}")
        
        if self.model_save_path and os.path.exists(self.model_save_path):
            state_dict = torch.load(self.model_save_path)
            self.infer_model = CompleteModel(model_dim, num_layers=config.num_layers).to(device)
            self.train_model = CompleteModel(model_dim, num_layers=config.num_layers).to(device)
            self.infer_model.load_state_dict(state_dict)
            print(f"Infer model loaded from {self.model_save_path}")
        else:
            self.infer_model = CompleteModel(model_dim, num_layers=config.num_layers).to(device)
            self.train_model = CompleteModel(model_dim, num_layers=config.num_layers).to(device)

        self.query_encoder = query_encoder
        self.loss_file = f'./final_results/loss{config.delta_weight}_layer{config.num_layers}_learnrate{config.learn_rate}_{config.num_epochs_per_train}per_train.txt'
        
        
        # Ensure both models have same initial parameters
        self.train_model.load_state_dict(self.infer_model.state_dict())
        
        # Shared memory buffer for parameter synchronization
        self.shared_state_dict = {}
        for name, param in self.train_model.state_dict().items():
            self.shared_state_dict[name] = param.share_memory_()
        
        self.sample_queue = mp.Queue(maxsize=config.max_sample_queue_size)
        self.batch_size = config.batch_size
        self.is_training = mp.Value('b', True)
        self.model_lock = mp.Lock()
        self.has_new_params = mp.Value('b', False)  # Flag for new parameters

        self.local_buffer = deque(maxlen=config.max_buffer_size)   # Buffer stores up to 5000 samples, adjustable based on memory
        self.num_epochs_per_train = config.num_epochs_per_train    # Number of epochs per training session
        self.train_batch_size = config.batch_size  # Mini-batch size
        
        # Add loss function
        self.criterion = CombinedLoss(delta_weight=config.delta_weight)
        # Add optimizer
        self.optimizer = torch.optim.Adam(self.train_model.parameters(), lr=config.learn_rate)

    def start_training_process(self):
        """Start training process"""
        self.training_process = mp.Process(
            target=self._training_loop,
            args=(self.sample_queue, self.is_training)
        )
        self.training_process.start()

    def _training_loop(self, sample_queue, is_training):
        """Training loop"""
        MONITOR_INTERVAL = 0.5  # Monitor interval in seconds
        
        while is_training.value:
            try:
                current_size = sample_queue.qsize()
                if current_size >= self.batch_size:
                    # Collect a batch of data
                    while True:
                        try:
                            sample = sample_queue.get_nowait()
                            self.local_buffer.append(sample)
                        except:
                            break

                    self._train_epochs(self.num_epochs_per_train)
                    with self.model_lock:
                        state_dict = self.train_model.state_dict()
                        for name, param in state_dict.items():
                            self.shared_state_dict[name].copy_(param)
                        self.has_new_params.value = True
                        print("Parameters synchronized to shared memory")
                else:
                    time.sleep(MONITOR_INTERVAL)
                    
            except Exception as e:
                print(f"Error in training loop: {str(e)}")
                traceback.print_exc()
                time.sleep(MONITOR_INTERVAL)

    def _train_epochs(self, num_epochs):
        buffer_list = list(self.local_buffer)
        data_size = len(buffer_list)
        
        for ep in range(num_epochs):
            np.random.shuffle(buffer_list)
            running_loss = 0.0
            for start_idx in range(0, data_size, self.train_batch_size):
                end_idx = start_idx + self.train_batch_size
                batch_data = buffer_list[start_idx:end_idx]
                
                if len(batch_data) < self.train_batch_size:
                    break

                predicates_matrices, join_masks, seq_lens, est_rows, act_rows = zip(*batch_data)
                max_seq_len = max(seq_len[0] for seq_len in seq_lens)  
                        
                padding_masks = []
                padded_predicates = []
                padded_joins = []
                
                # Process each sample
                for pred_mat, join_mat, seq_len in zip(predicates_matrices, join_masks, seq_lens):
                    curr_len = seq_len[0]
                    # Create padding mask: True for positions to mask (padding), False for valid positions
                    mask = torch.zeros((1, 1, curr_len), dtype=torch.bool)  # Changed to boolean type
                    
                    if curr_len < max_seq_len:
                        # Add padding mask, set padding positions to True
                        mask = torch.cat([
                            mask,
                            torch.ones((1, 1, max_seq_len - curr_len), dtype=torch.bool)  # Padding positions are True
                        ], dim=2)
                        # Add padding to predicates matrix
                        pred_pad = torch.zeros((1, max_seq_len - curr_len, pred_mat.size(-1)))
                        pred_mat = torch.cat([pred_mat, pred_pad], dim=1)
                        # Add padding to join mask
                        join_pad_rows = torch.zeros((1, max_seq_len - curr_len, join_mat.size(-1)))
                        join_mat = torch.cat([join_mat, join_pad_rows], dim=1)
                        join_pad_cols = torch.zeros((1, max_seq_len, max_seq_len - curr_len))
                        join_mat = torch.cat([join_mat, join_pad_cols], dim=2)
                    
                    padding_masks.append(mask)
                    padded_predicates.append(pred_mat)
                    # print(f"Padded predicates: {pred_mat.shape}")
                    padded_joins.append(join_mat)
                
                predicates_batch = torch.cat(padded_predicates, dim=0)  # [batch_size, seq_len, feature_dim]
                join_mask_batch = torch.cat(padded_joins, dim=0)  # [batch_size, seq_len, seq_len]
                padding_mask_batch = torch.cat(padding_masks, dim=0)  # [batch_size, 1, seq_len]
                est_rows_batch = torch.cat(est_rows, dim=0)
                act_rows_batch = torch.cat(act_rows, dim=0)
                
                predicates_batch = predicates_batch.to(device)
                join_mask_batch = join_mask_batch.to(device)
                padding_mask_batch = padding_mask_batch.to(device)
                est_rows_batch = est_rows_batch.to(device)
                act_rows_batch = act_rows_batch.to(device)
                
                self.train_model.train()
                self.optimizer.zero_grad()
                
                prediction, delta, _ = self.train_model(
                    x=predicates_batch,
                    initial_estimate=est_rows_batch,
                    padding_mask=padding_mask_batch,
                    attention_mask=join_mask_batch,
                    seq_lengths=[s[0] for s in seq_lens]
                )
                
                loss = self.criterion(prediction, act_rows_batch, delta)
                
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / (data_size / self.train_batch_size)
            with open(self.loss_file, "a") as f:
                f.write(f"[Epoch {ep+1}/{num_epochs}] avg_loss: {avg_loss:.4f}\n")

        

    def infer(self, subquery: str):
        """Inference interface"""
        # Check and update parameters
        with self.model_lock:
            if self.has_new_params.value:  # If there are new parameter updates
                state_dict = {name: param.clone() for name, param in self.shared_state_dict.items()}
                print(f"update infer model parameters")
                self.infer_model.load_state_dict(state_dict)
                self.has_new_params.value = False  # Reset flag
        
        table_aliases, nrows = get_table_aliases(subquery)
        predicates_matrix, all_predicates, join_mask = self.query_encoder.encode_subquery(table_aliases)
        
        # Add batch dimension
        seq_len = []
        predicates_matrix = torch.from_numpy(predicates_matrix).float().unsqueeze(0)  # (16,17) -> (1,16,17)
        join_mask = torch.from_numpy(join_mask).float().unsqueeze(0)  # (N,N) -> (1,N,N)
        nrows = torch.tensor([nrows], dtype=torch.float32).unsqueeze(0)  # Single value -> (1,)
        seq_len = [predicates_matrix.size(1)]
        
        self.infer_model.eval()
        with torch.no_grad():
            prediction, delta, attentions = self.infer_model(predicates_matrix, nrows, None, join_mask, seq_len)
            prediction_value = round(prediction.item())
            delta_value = round(delta.item())
            return prediction_value, attentions, all_predicates
        
    def infer_for_mscn(self, subquery: str):
        """Inference interface"""
        with self.model_lock:
            if self.has_new_params.value:  # If there are new parameter updates
                state_dict = {name: param.clone() for name, param in self.shared_state_dict.items()}
                print(f"update infer model parameters")
                self.infer_model.load_state_dict(state_dict)
                self.has_new_params.value = False  # Reset flag

        table_aliases, nrows = get_table_aliases(subquery)
        table_matrix, join_matrix, predicate_matrix = self.query_encoder.encode_subquery_for_mscn(table_aliases)
        table_matrix = torch.from_numpy(table_matrix).float().unsqueeze(0) 
        join_matrix = torch.from_numpy(join_matrix).float().unsqueeze(0)  
        predicate_matrix = torch.from_numpy(predicate_matrix).float().unsqueeze(0)  
        nrows = torch.tensor([nrows], dtype=torch.float32).unsqueeze(0)  
        self.infer_model.eval()
        with torch.no_grad():
            prediction, delta = self.infer_model(table_matrix, nrows, None, join_matrix)
            prediction_value = round(prediction.item())
            delta_value = round(delta.item())

    def add_training_sample(self, sample):
        """Add training sample to queue"""
        tables, est_rows, act_rows = sample
        predicates_matrix, _, join_mask = self.query_encoder.encode_subquery(tables)
        predicates_matrix = torch.from_numpy(predicates_matrix).float().unsqueeze(0)  # (16,17) -> (1,16,17)
        join_mask = torch.from_numpy(join_mask).float().unsqueeze(0)  # (N,N) -> (1,N,N)
        est_rows = torch.tensor([est_rows], dtype=torch.float32).unsqueeze(0)  
        act_rows = torch.tensor([act_rows], dtype=torch.float32).unsqueeze(0)  
        seq_len = [predicates_matrix.size(1)]
        self.sample_queue.put((predicates_matrix, join_mask, seq_len, est_rows, act_rows))

    def cleanup(self):
        """Clean up resources"""
        self.is_training.value = False
        if hasattr(self, 'training_process'):
            self.training_process.join()
        # Empty sample queue
        while not self.sample_queue.empty():
            self.sample_queue.get()

    def update_query_feature(self, query_feature: QueryFeature):
        """Update query feature"""
        self.query_encoder.update_query_feature(query_feature)

    def encode_subquery(self, table_aliases: List[str]) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Encode subquery and generate join mask"""
        return self.query_encoder.encode_subquery(table_aliases)
    
    def save_infer_model(self):
        """Save inference model to file"""
        if self.model_save_path:
            os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
            torch.save(self.infer_model.state_dict(), self.model_save_path)
            print(f"Infer model saved to {self.model_save_path}")

    def load_infer_model(self):
        """Load inference model from file"""
        if os.path.exists(self.model_save_path):
            self.infer_model.load_state_dict(torch.load(self.model_save_path))
            self.train_model.load_state_dict(torch.load(self.model_save_path))
            print(f"Infer model loaded from {self.model_save_path}")
        else:
            print(f"No model found at {self.model_save_path}, loading skipped")
