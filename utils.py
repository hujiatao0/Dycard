import math
from typing import List, Tuple
from pg_executor import get_database_schema, TableEmbedder, ColumnEmbedder, load_embedders, save_embedders
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_model_dim(num_tables, num_columns):
    table_embedding_dim = (max(int(math.sqrt(num_tables)) + 1, 3))
    column_embedding_dim = (int(math.sqrt(num_columns)) + 1)
    return table_embedding_dim + column_embedding_dim + 6 + 1


def initialize_database_embeddings(save_path=None):
    print("Initializing database schema and embeddings...")
    
    # Get database stats
    column_stats = get_database_schema(save_path)
    if not column_stats:
        raise Exception("Failed to get database schema")
    
    # Count tables and columns
    tables = set()
    columns = set()
    for key in column_stats.stats_dict:
        table_name, column_name = key.split('.')
        tables.add(table_name)
        columns.add(key)
    
    total_tables = len(tables)
    total_columns = len(columns)
    
    print(f"Found {total_tables} tables and {total_columns} columns")
    
    table_embedder, column_embedder = None, None
    if save_path:
        table_embedder, column_embedder = load_embedders(save_path, total_tables, total_columns)
    
    if table_embedder is None or column_embedder is None:
        # column_embedding_dim = int(math.sqrt(total_columns) + 1)
        # table_embedding_dim = max(int(math.sqrt(total_tables)) + 1, 3)
        column_embedding_dim = int(math.sqrt(total_columns) + 1)
        table_embedding_dim = max(int(math.sqrt(total_tables)) + 1, 3)
        print(f"Using embedding dimensions: {table_embedding_dim} for tables, {column_embedding_dim} for columns")
        
        table_embedder = TableEmbedder(total_tables, table_embedding_dim)
        column_embedder = ColumnEmbedder(total_columns, column_embedding_dim)
        
        for key in column_stats.stats_dict:
            table_name, column_name = key.split('.')
            table_embedder.register_table(table_name)
            column_embedder.register_column(table_name, column_name)
        
        if save_path:
            save_embedders(table_embedder, column_embedder, save_path)
    
    print("\nDatabase embeddings initialized successfully")
    return table_embedder, column_embedder, column_stats


def get_table_aliases(subquery: str) -> Tuple[List[str], int]:
    """
    Parse subquery string and return table alias list and row count
    
    Args:
        subquery: String like "join badges and users | rows: 6472529" or "table pl | rows: 6901"
    
    Returns:
        Tuple[List[str], int]: List of table aliases and row count
    """
    # Split string to get table part and row count part
    parts = subquery.split('|')
    if len(parts) != 2:
        raise ValueError(f"Invalid subquery format: {subquery}")
    
    # Parse row count
    rows_match = re.search(r'rows:\s*(\d+)', parts[1])
    if not rows_match:
        raise ValueError(f"Cannot find rows in: {parts[1]}")
    rows = int(rows_match.group(1))
    
    # Parse table aliases
    table_part = parts[0].strip()
    if table_part.startswith('join'):
        # Multiple tables case: join badges and users
        aliases = re.findall(r'\b([a-zA-Z][a-zA-Z0-9_]*)\b(?!\s*:)', table_part)
    else:
        # Single table case: table pl
        alias_match = re.search(r'table\s+([a-zA-Z][a-zA-Z0-9_]*)\b', table_part)
        if not alias_match:
            raise ValueError(f"Cannot find table alias in: {table_part}")
        aliases = [alias_match.group(1)]
    
    return aliases, rows

def extract_training_samples(plan_node):
    """Recursively extract training samples
    Returns:
        List[Tuple[List[str], int, int]]: Each sample contains (table list, estimated_rows, actual_rows)
    """
    samples = []
    
    # Check if current node meets conditions
    node_type = plan_node.get('node_type', '').lower()
    if ('scan' in node_type or 'join' in node_type or 'loop' in node_type) and not 'index' in node_type.lower():
        tables = plan_node.get('tables', [])
        estimated_rows = plan_node.get('estimated_rows')
        actual_rows = plan_node.get('actual_rows')
        if tables and estimated_rows is not None and actual_rows is not None:
            # Directly construct sample using table list and row counts
            samples.append((tables, estimated_rows, actual_rows))
    
    # Recursively process child nodes
    for child in plan_node.get('children', []):
        samples.extend(extract_training_samples(child))
    
    return samples


def visualize_attention_correlations(attention_list, predicate_list, save_prefix=""):
    """
    Visualize correlations between multiple layers of attention and generate heatmaps for each layer's attention.
    Uses predicate_list as x/y axis labels for intuitive naming of sequence positions.

    Args:
        attention_list: list of torch.Tensor
            List of attention weights, length = block_num,
            each element shape [1, seq_len, seq_len].
        predicate_list: list of str
            Human-readable names for each position (token/predicate) in sequence, matching seq_len of attention matrices.
        save_prefix: str
            Optional prefix for saved image filenames.
    """
    # Basic checks
    block_num = len(attention_list)
    if block_num == 0:
        print("[WARN] attention_list is empty, cannot visualize.")
        return
    
    # Get seq_len from first attention matrix and verify alignment with predicate_list
    # Assuming shape is [1, seq_len, seq_len]
    seq_len = attention_list[0].shape[-1]
    if len(predicate_list) != seq_len:
        raise ValueError(f"predicate_list length ({len(predicate_list)}) "
                         f"does not match attention matrix seq_len ({seq_len})!")
    
    # Convert [1, seq_len, seq_len] --> [seq_len, seq_len] and to numpy
    attn_mats = []
    for attn in attention_list:
        attn_2d = attn.squeeze(0).detach().cpu().numpy()  # [seq_len, seq_len]
        attn_mats.append(attn_2d)
    
    # Calculate correlation coefficients between pairs of attention matrices
    attn_vectors = [mat.flatten() for mat in attn_mats]  # each [seq_len * seq_len]
    corr_matrix = np.zeros((block_num, block_num))

    for i in range(block_num):
        for j in range(block_num):
            corr = np.corrcoef(attn_vectors[i], attn_vectors[j])[0, 1]
            corr_matrix[i, j] = corr

    # (A) First plot correlation heatmap between layers
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix, annot=True, cmap="Blues", vmin=-1, vmax=1,
        xticklabels=[f"Block {i+1}" for i in range(block_num)],
        yticklabels=[f"Block {i+1}" for i in range(block_num)]
    )
    plt.title("Correlation Between Blocks' Attention Maps")
    plt.tight_layout()
    corr_save_path = f"{save_prefix}correlation_heatmap.png"
    plt.savefig(corr_save_path)
    plt.close()
    print(f"[INFO] Correlation heatmap saved to: {corr_save_path}")

    # (B) Plot and save attention distribution for each layer
    #     Set X/Y axis tick labels to predicate_list
    for i, mat in enumerate(attn_mats):
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            mat,
            cmap="Blues",
            xticklabels=predicate_list,
            yticklabels=predicate_list
        )
        plt.title(f"Block {i+1} Attention")
        # Rotate X axis labels if needed to prevent overlap
        plt.xticks(rotation=70)
        plt.yticks(rotation=0)
        plt.tight_layout()
        block_save_path = f"{save_prefix}block_{i+1}_attention.png"
        plt.savefig(block_save_path)
        plt.close()
        print(f"[INFO] Block {i+1} attention heatmap saved to: {block_save_path}")


# Test cases
if __name__ == "__main__":
    test_cases = [
        "table pl | rows: 6901",
        "table badges | rows: 100247",
        "join badges and users | rows: 6472529",
        "join posthistory and votes | rows: 5000",
        "join pl and postlinks | rows: 3000",
        "join badges and users and votes | rows: 8000"
    ]
    
    for test in test_cases:
        aliases, rows = get_table_aliases(test)
        print(f"Input: {test}")
        print(f"Output: aliases={aliases}, rows={rows}")
        print("-" * 50)