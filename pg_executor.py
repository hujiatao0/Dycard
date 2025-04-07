import psycopg2
from psycopg2 import Error
import json
import numpy as np
import torch
import torch.nn as nn
import math
import os
from decimal import Decimal
import warnings
from config import config

warnings.simplefilter('ignore', FutureWarning)

class TableEmbedder(nn.Module):
    def __init__(self, num_tables, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_tables, embedding_dim)
        self.table_to_idx = {}
        self.idx_to_table = {}
        self.current_idx = 0
    
    def register_table(self, table_name):
        if table_name not in self.table_to_idx:
            self.table_to_idx[table_name] = self.current_idx
            self.idx_to_table[self.current_idx] = table_name
            self.current_idx += 1
    
    def get_embedding(self, table_name):
        idx = self.table_to_idx.get(table_name)
        if idx is not None:
            embedding = self.embedding(torch.tensor(idx))
            return embedding.detach().numpy().tolist()  # Convert to list for JSON serialization
        return None

    def get_all_embeddings(self):
        return {table_name: self.get_embedding(table_name) 
                for table_name in self.table_to_idx}

class ColumnEmbedder(nn.Module):
    def __init__(self, num_columns, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_columns + 1, embedding_dim)
        self.column_to_idx = {}
        self.idx_to_column = {}
        self.current_idx = 0
        self.unknown_idx = num_columns
    
    def register_column(self, table_name, column_name):
        key = f"{table_name}.{column_name}"
        if key not in self.column_to_idx:
            self.column_to_idx[key] = self.current_idx
            self.idx_to_column[self.current_idx] = key
            self.current_idx += 1
    
    def get_embedding(self, table_name, column_name):
        key = f"{table_name}.{column_name}"
        idx = self.column_to_idx.get(key, self.unknown_idx)
        embedding = self.embedding(torch.tensor(idx))
        return embedding.detach().numpy().tolist()  # Convert to list for JSON serialization
    
    def get_unknown_embedding(self):
        embedding = self.embedding(torch.tensor(self.unknown_idx))
        return embedding.detach().numpy().tolist()

    def get_all_embeddings(self):
        return {col_key: self.get_embedding(*col_key.split('.'))
                for col_key in self.column_to_idx}

class ColumnStats:
    def __init__(self):
        self.stats_dict = {}
    
    def add_stats(self, table_name, column_name, stats):
        key = f"{table_name}.{column_name}"
        self.stats_dict[key] = stats
    
    def get_stats(self, table_name, column_name):
        key = f"{table_name}.{column_name}"
        return self.stats_dict.get(key)
    
    def save_stats(self, save_path):
        """Save stats to JSON file"""
        os.makedirs(save_path, exist_ok=True)
        stats_file = os.path.join(save_path, 'column_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats_dict, f, indent=2, default=decimal_default)
        print(f"Column stats saved to: {stats_file}")
    
    def load_stats(self, save_path):
        """Load stats from JSON file"""
        stats_file = os.path.join(save_path, 'column_stats.json')
        if os.path.exists(stats_file):
            with open(stats_file, 'r', encoding='utf-8') as f:
                self.stats_dict = json.load(f)
            print("Column stats loaded from saved file")
            return True
        return False

def decimal_default(obj):
    """Helper function to convert Decimal objects to strings for JSON serialization"""
    if isinstance(obj, Decimal):
        return str(obj)
    raise TypeError

def get_column_stats(cursor, table_name, column_name, data_type):
    numeric_types = ['integer', 'numeric', 'decimal', 'real', 'double precision', 'smallint', 'bigint']
    
    if any(t in data_type.lower() for t in numeric_types):
        try:
            if 'bigint' in data_type.lower():
                cursor.execute(f"""
                    SELECT 
                        MIN({column_name})::text,
                        MAX({column_name})::text,
                        COUNT({column_name})
                    FROM {table_name}
                """)
                min_val, max_val, count = cursor.fetchone()
                return {
                    'min': str(min_val) if min_val is not None else None,
                    'max': str(max_val) if max_val is not None else None,
                    'count': int(count) if count is not None else None,
                    'type': 'bigint'
                }
            else:
                cursor.execute(f"""
                    SELECT 
                        MIN({column_name}),
                        MAX({column_name}),
                        ROUND(AVG({column_name})::numeric, 2),
                        COUNT({column_name})
                    FROM {table_name}
                """)
                min_val, max_val, avg_val, count = cursor.fetchone()
                return {
                    'min': str(min_val) if min_val is not None else None,
                    'max': str(max_val) if max_val is not None else None,
                    'avg': str(avg_val) if avg_val is not None else None,
                    'count': int(count) if count is not None else None,
                    'type': 'numeric'
                }
        except Exception as e:
            print(f"Error getting stats for column {table_name}.{column_name}: {e}")
            return None
    return None

def get_database_schema(save_path=None):
    try:
        column_stats = ColumnStats()
        if save_path and column_stats.load_stats(save_path):
            return column_stats
        
        connection = psycopg2.connect(
            user=config.pg_user,
            host="localhost",
            port=config.pg_port,
            database= config.dbname
        )
        
        cursor = connection.cursor()
        
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = cursor.fetchall()
        
        print("Database tables:")
        for table in tables:
            table_name = table[0]
            print(f"\nTable name: {table_name}")
            
            cursor.execute(f"""
                SELECT column_name, data_type, character_maximum_length
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
            """)
            columns = cursor.fetchall()
            
            print("Column information:")
            for column in columns:
                column_name, data_type, max_length = column
                stats = get_column_stats(cursor, table_name, column_name, data_type)
                if stats:
                    column_stats.add_stats(table_name, column_name, stats)
                    if stats.get('type') == 'bigint':
                        print(f"- {column_name} ({data_type})")
                        print(f"  Stats: min={stats['min']}, max={stats['max']}, count={stats['count']}")
                    else:
                        print(f"- {column_name} ({data_type})")
                        print(f"  Stats: min={stats['min']}, max={stats['max']}, avg={stats['avg']}, count={stats['count']}")
        
        if save_path:
            column_stats.save_stats(save_path)
        
        return column_stats

    except (Exception, Error) as error:
        print(f"Error connecting to database: {error}")
        return None
        
    finally:
        if 'connection' in locals():
            cursor.close()
            connection.close()
            print("\nDatabase connection closed")

def save_embedders(table_embedder, column_embedder, save_path):
    """Save embedders using PyTorch's save method"""
    os.makedirs(save_path, exist_ok=True)
    
    table_state = {
        'model_state': table_embedder.state_dict(),
        'table_to_idx': table_embedder.table_to_idx,
        'idx_to_table': table_embedder.idx_to_table,
        'current_idx': table_embedder.current_idx
    }
    torch.save(table_state, os.path.join(save_path, 'table_embedder.pt'))
    
    column_state = {
        'model_state': column_embedder.state_dict(),
        'column_to_idx': column_embedder.column_to_idx,
        'idx_to_column': column_embedder.idx_to_column,
        'current_idx': column_embedder.current_idx,
        'unknown_idx': column_embedder.unknown_idx
    }
    torch.save(column_state, os.path.join(save_path, 'column_embedder.pt'))
    print("Embedders saved successfully")

def load_embedders(save_path, num_tables, num_columns):
    """Load embedders using PyTorch's load method"""
    table_embedder = None
    column_embedder = None
    
    table_path = os.path.join(save_path, 'table_embedder.pt')
    column_path = os.path.join(save_path, 'column_embedder.pt')
    
    if os.path.exists(table_path) and os.path.exists(column_path):
        table_state = torch.load(table_path)
        # table_embedder = TableEmbedder(num_tables, max(int(math.sqrt(num_tables)) + 1, 3))
        table_embedder = TableEmbedder(num_tables, max(int(math.sqrt(num_tables)) + 1, 3))
        table_embedder.load_state_dict(table_state['model_state'])
        table_embedder.table_to_idx = table_state['table_to_idx']
        table_embedder.idx_to_table = table_state['idx_to_table']
        table_embedder.current_idx = table_state['current_idx']
        
        column_state = torch.load(column_path)
        # column_embedder = ColumnEmbedder(num_columns, int(math.sqrt(num_columns)) + 1)
        column_embedder = ColumnEmbedder(num_columns, (int(math.sqrt(num_columns)) + 1))
        column_embedder.load_state_dict(column_state['model_state'])
        column_embedder.column_to_idx = column_state['column_to_idx']
        column_embedder.idx_to_column = column_state['idx_to_column']
        column_embedder.current_idx = column_state['current_idx']
        column_embedder.unknown_idx = column_state['unknown_idx']
        
        print("Embedders loaded successfully")
    
    return table_embedder, column_embedder

def main():
    column_stats = get_database_schema()
    if not column_stats:
        return
    
    tables = set()
    columns = set()
    for key in column_stats.stats_dict:
        table_name, column_name = key.split('.')
        tables.add(table_name)
        columns.add(key)
    
    total_tables = len(tables)
    total_columns = len(columns)
    
    column_embedding_dim = (int(math.sqrt(total_columns)) + 1)
    table_embedding_dim = (max(int(math.sqrt(total_tables)) + 1, 3))
    
    table_embedder = TableEmbedder(total_tables, table_embedding_dim)
    column_embedder = ColumnEmbedder(total_columns, column_embedding_dim)
    
    for key in column_stats.stats_dict:
        table_name, column_name = key.split('.')
        print(f"Registering table: {table_name}, column: {column_name}")
        table_embedder.register_table(table_name)
        column_embedder.register_column(table_name, column_name)

if __name__ == "__main__":
    main()


