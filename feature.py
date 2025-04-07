from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import re
import numpy as np
import torch
from pg_executor import TableEmbedder, ColumnEmbedder, ColumnStats
from utils import initialize_database_embeddings
from config import config

@dataclass
class TableInfo:
    """Table information class"""
    name: str  # Original table name
    alias: str  # Table alias
    predicates: List[str]  # List of predicate conditions for this table

@dataclass
class JoinInfo:
    """Join information class"""
    table1: str  # Table 1 (using alias)
    column1: str  # Column of table 1
    operator: str  # Join operator (=, <, >, <=, >=, !=)
    table2: str  # Table 2 (using alias)
    column2: str  # Column of table 2

class QueryFeature:
    def __init__(self, query: str):
        self.query = self._clean_query(query)
        self.tables: Dict[str, TableInfo] = {}  # Mapping from alias to TableInfo
        self.name_to_alias: Dict[str, str] = {}  # Mapping from original table name to alias
        self.join_conditions: List[JoinInfo] = []  # Store all join conditions
        self.table_predicates: Dict[str, List[str]] = {}  # Mapping from table alias to predicates
        
        # Parse query
        self._parse_query()
    
    def _clean_query(self, query: str) -> str:
        """Clean query string"""
        # Remove extra whitespace
        query = ' '.join(query.split())
        
        # Remove EXPLAIN related prefixes (case insensitive)
        prefixes_to_remove = [
            'explain analyze verbose',
            'explain analyze',
            'explain (analyze, verbose)',
            'explain (analyze)',
            'explain verbose',
            'explain'
        ]
        
        query_lower = query.lower()
        for prefix in prefixes_to_remove:
            if query_lower.startswith(prefix):
                query = query[len(prefix):].strip()
                break
        
        # Remove trailing semicolon
        return query.rstrip(';')
    
    def _parse_query(self):
        """Parse SQL query"""
        # Extract table info between FROM and WHERE
        from_where_pattern = r'from\s+(.*?)\s+where\s+(.*)'
        match = re.search(from_where_pattern, self.query, re.IGNORECASE)
        if match:
            tables_str, conditions_str = match.groups()
            self._extract_tables(tables_str)
            self._process_conditions(conditions_str)
    
    def _extract_tables(self, tables_str: str):
        """Extract table information"""
        # Split table list
        tables = [t.strip() for t in tables_str.split(',')]
        
        for table in tables:
            # Handle potential AS keyword (case insensitive)
            parts = re.split(r'\s+(?:as\s+)?', table.strip(), flags=re.IGNORECASE)
            if len(parts) >= 2:
                table_name = parts[0].strip()
                alias = parts[-1].strip()
            else:
                table_name = alias = parts[0].strip()
            
            self.tables[alias] = TableInfo(name=table_name, alias=alias, predicates=[])
            self.name_to_alias[table_name] = alias
            self.table_predicates[alias] = []
    
    def _process_conditions(self, conditions_str: str):
        """Process all conditions in WHERE clause"""
        conditions = [c.strip() for c in conditions_str.split('and')]
        
        for condition in conditions:
            # Check if it's a join condition (contains two table aliases)
            if sum(t + '.' in condition for t in self.tables) == 2:
                self._process_join_condition(condition)
            else:
                self._process_predicate(condition)
    
    def _process_join_condition(self, condition: str):
        """Process join condition"""
        # Find operator
        operators_pattern = r'(>=|<=|!=|=|>|<)'
        parts = re.split(operators_pattern, condition)
        if len(parts) == 3:
            left, operator, right = [p.strip() for p in parts]
            
            left_parts = left.split('.')
            right_parts = right.split('.')
            
            if len(left_parts) == 2 and len(right_parts) == 2:
                join_info = JoinInfo(
                    table1=left_parts[0],
                    column1=left_parts[1],
                    operator=operator,
                    table2=right_parts[0],
                    column2=right_parts[1]
                )
                self.join_conditions.append(join_info)
    
    def _process_predicate(self, condition: str):
        """Process predicate condition"""
        condition = condition.strip()
        # Find table alias involved in condition
        for alias in self.tables:
            if f"{alias}." in condition:
                self.table_predicates[alias].append(condition)
                self.tables[alias].predicates.append(condition)
                break
    
    def get_table_joins(self, table_aliases: List[str]) -> List[JoinInfo]:
        """Get join conditions between specified tables"""
        return [
            join for join in self.join_conditions 
            if join.table1 in table_aliases and join.table2 in table_aliases
        ]
    
    def get_table_predicates(self, table_alias: str) -> List[str]:
        """Get all predicate conditions for specified table"""
        return self.table_predicates.get(table_alias, [])
    
    def get_original_table_name(self, alias: str) -> str:
        """Get original table name from alias"""
        return self.tables[alias].name if alias in self.tables else None

class SubQueryEncoder:
    def __init__(self, table_embedder: TableEmbedder, column_embedder: ColumnEmbedder, 
                 column_stats: ColumnStats, query_feature: QueryFeature):
        self.table_embedder = table_embedder
        self.column_embedder = column_embedder
        self.column_stats = column_stats
        self.query_feature = query_feature
        # Define one-hot encoding for operators
        self.operators = ['=', '>', '<', '>=', '<=', '!=']
        self.operator_to_idx = {op: idx for idx, op in enumerate(self.operators)}

        self.mask_values = {
            'no_join': 0.1,    # No join relationship
            'equal': 0.8,      # Equal join
            'not_equal': 0.4,  # Not equal relationship
            'greater': 0.6,    # Greater than relationship
            'less': -0.6,      # Less than relationship
            'self': 1.0        # Diagonal
        }
        
        # Define mapping for comparison operators
        self.compare_ops = {
            '>': 'greater',
            '>=': 'greater',
            '<': 'less',
            '<=': 'less',
            '=': 'equal',
            '!=': 'not_equal'
        }
        
    def _encode_operator(self, operator: str) -> np.ndarray:
        """Convert operator to one-hot encoding"""
        encoding = np.zeros(len(self.operators))
        if operator in self.operator_to_idx:
            encoding[self.operator_to_idx[operator]] = 1
        return encoding
    
    def _create_join_mask(self, predicates_info: List[Tuple[str, str, str]], 
                         join_conditions: List[JoinInfo]) -> np.ndarray:
        """
        Create join mask matrix
        Args:
            predicates_info: List of (table_alias, column_name, predicate) tuples
            join_conditions: List of JoinInfo objects
        Returns:
            L x L mask matrix
        """
        L = len(predicates_info)
        # print(predicates_info)
        # print(join_conditions)  
        mask = np.full((L, L), self.mask_values['no_join'])  # Default to no_join
        
        # Set diagonal
        np.fill_diagonal(mask, self.mask_values['self'])
        
        # Process join conditions
        for i in range(L):
            table_i, col_i, _ = predicates_info[i]
            for j in range(L):
                table_j, col_j, _ = predicates_info[j]
                if (table_i == table_j and i != j):
                    mask[i, j] = self.mask_values['equal']
                    mask[j, i] = mask[i, j]
                    continue
                
                # Check if join exists between these two columns
                for join in join_conditions:
                    # Check forward join
                    if (table_i == join.table1 and col_i == join.column1 and 
                        table_j == join.table2 and col_j == join.column2):
                        mask_type = self.compare_ops[join.operator]
                        mask[i, j] = self.mask_values[mask_type]
                        if mask_type in ['greater', 'less']:
                            mask[j, i] = -mask[i, j]  # Reverse sign for symmetric position
                        else:
                            mask[j, i] = mask[i, j]  # Same value for equal and not equal joins
                        break
                        
                    # Check reverse join
                    elif (table_i == join.table2 and col_i == join.column2 and 
                          table_j == join.table1 and col_j == join.column1):
                        mask_type = self.compare_ops[join.operator]
                        if mask_type in ['greater', 'less']:
                            mask_type = 'less' if mask_type == 'greater' else 'greater'
                        mask[i, j] = self.mask_values[mask_type]
                        if mask_type in ['greater', 'less']:
                            mask[j, i] = -mask[i, j]
                        else:
                            mask[j, i] = mask[i, j]
                        break
        return mask
    
    def _get_default_encoding(self, table_name: str) -> np.ndarray:
        """Get default encoding for table (used when no predicates exist)"""
        # Get table embedding
        table_emb = np.array(self.table_embedder.get_embedding(table_name))
        # Get unknown column embedding
        unknown_column_emb = np.array(self.column_embedder.get_unknown_embedding())
        # Zero operator encoding
        operator_emb = np.zeros(len(self.operators))
        # Zero value
        value_emb = np.array([0.0])
        return np.concatenate([table_emb, unknown_column_emb, operator_emb, value_emb])

    def _encode_join_column(self, table_name: str, column_name: str) -> np.ndarray:
        """Encode join column (using actual table and column embeddings, zero operator and value)"""
        # Get table embedding
        table_emb = np.array(self.table_embedder.get_embedding(table_name))
        # Get column embedding
        column_emb = np.array(self.column_embedder.get_embedding(table_name, column_name))
        # Zero operator encoding
        operator_emb = np.zeros(len(self.operators))
        # Zero value
        value_emb = np.array([0.0])
            
        return np.concatenate([table_emb, column_emb, operator_emb, value_emb])
    
    def _normalize_value(self, value: float, table_name: str, column_name: str) -> float:
        """Normalize value to 0-1 range based on column statistics"""
        stats = self.column_stats.get_stats(table_name, column_name)
        if stats and stats['min'] is not None and stats['max'] is not None:
            min_val = float(stats['min'])
            max_val = float(stats['max'])
            if max_val > min_val:
                return (float(value) - min_val) / (max_val - min_val)
        return 0.0
    
    def encode_predicate(self, predicate: str, table_alias: str, 
                    table_name: str) -> np.ndarray:
        """Encode single predicate condition"""
        # Break down predicate
        operators_pattern = r'(>=|<=|!=|=|>|<)'
        parts = re.split(operators_pattern, predicate)
        if len(parts) != 3:
            return None
                
        column_part, operator, value = [p.strip() for p in parts]
        column_name = column_part.split('.')[1]  # Get column name
            
        # Get encodings for each part
        table_emb = np.array(self.table_embedder.get_embedding(table_name))
        column_emb = np.array(self.column_embedder.get_embedding(table_name, column_name))
        operator_emb = self._encode_operator(operator)
        value_emb = np.array([self._normalize_value(float(value), table_name, column_name)])
            
        # Concatenate all parts
        return np.concatenate([table_emb, column_emb, operator_emb, value_emb])
    
    def update_query_feature(self, query_feature: QueryFeature):
        self.query_feature = query_feature
    
    def encode_subquery(self, table_aliases: List[str]) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Encode subquery and generate join mask"""
        all_predicates = []
        encoded_predicates = []
        predicates_info = []  # Store (table_alias, column_name, predicate) info
        
        # Get relevant join conditions
        join_conditions = self.query_feature.get_table_joins(table_aliases)
        
        # Generate encoding for each table
        for alias in table_aliases:
            table_name = self.query_feature.get_original_table_name(alias)
            if table_name:
                # First handle join columns
                for join in join_conditions:
                    if alias == join.table1:
                        encoded = self._encode_join_column(table_name, join.column1)
                        encoded_predicates.append(encoded)
                        all_predicates.append(f"{alias}.{join.column1} (join column)")
                        predicates_info.append((alias, join.column1, f"{alias}.{join.column1} (join column)"))
                    if alias == join.table2:
                        encoded = self._encode_join_column(table_name, join.column2)
                        encoded_predicates.append(encoded)
                        all_predicates.append(f"{alias}.{join.column2} (join column)")
                        predicates_info.append((alias, join.column2, f"{alias}.{join.column2} (join column)"))
                
                # Then handle predicate columns
                predicates = self.query_feature.get_table_predicates(alias)
                if predicates:
                    for pred in predicates:
                        encoded = self.encode_predicate(pred, alias, table_name)
                        if encoded is not None:
                            encoded_predicates.append(encoded)
                            all_predicates.append(pred)
                            operators_pattern = r'(>=|<=|!=|=|>|<)'
                            parts = re.split(operators_pattern, pred)
                            column_name = parts[0].split('.')[1].strip()
                            predicates_info.append((alias, column_name, pred))
                
                # Add default encoding if table has no joins or predicates
                if not predicates and not any(alias in (j.table1, j.table2) for j in join_conditions):
                    default_encoding = self._get_default_encoding(table_name)
                    encoded_predicates.append(default_encoding)
                    all_predicates.append(f"{alias} (no predicates)")
                    predicates_info.append((alias, "unknown", f"{alias} (no predicates)"))
        
        # Generate predicates matrix
        predicates_matrix = np.stack(encoded_predicates)
        
        # Generate join mask
        join_mask = self._create_join_mask(predicates_info, join_conditions)
        
        return predicates_matrix, all_predicates, join_mask
    
    def _encode_join_column_for_mscn(self, table_alias: str, column_name: str) -> np.ndarray:
        """Encode join column for MSCN (using actual table and column embeddings)"""
        # Get table embedding
        table_name = self.query_feature.get_original_table_name(table_alias)
        table_emb = np.array(self.table_embedder.get_embedding(table_name))
        # Get column embedding
        column_emb = np.array(self.column_embedder.get_embedding(table_name, column_name))
        return np.concatenate([table_emb, column_emb])
    
    def encode_subquery_for_mscn(self, table_aliases: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode subquery and generate join mask for MSCN"""
        join_encoded_list = []
        table_encoded_list = []
        predicate_encoded_list = []
        
        join_conditions = self.query_feature.get_table_joins(table_aliases)

        for join in join_conditions:
            encoded1 = self._encode_join_column_for_mscn(join.table1, join.column1)
            encoded2 = self._encode_join_column_for_mscn(join.table2, join.column2)
            # use operator one-hot encoding
            operator_vec = self._encode_operator(join.operator)
            concat_encoded = np.concatenate([encoded1, operator_vec, encoded2])
            join_encoded_list.append(concat_encoded)

        for alias in table_aliases:
            table_name = self.query_feature.get_original_table_name(alias)
            table_emb = np.array(self.table_embedder.get_embedding(table_name))
            table_encoded_list.append(table_emb)

        for alias in table_aliases:
            predicates = self.query_feature.get_table_predicates(alias)
            if predicates:
                for pred in predicates:
                    encoded = self.encode_predicate(pred, alias, table_name)
                    predicate_encoded_list.append(encoded)

        predicates_matrix = np.stack(predicate_encoded_list)
        table_matrix = np.stack(table_encoded_list)
        join_matrix = np.stack(join_encoded_list)
        
        return table_matrix, join_matrix, predicates_matrix
        
        

# Example query
def main():
    query = """
    select count(*) from badges as b, comments as c, postlinks as pl, posts as p, 
    users as u, votes as v where c.postid = v.postid and p.owneruserid = b.userid 
    and p.id = c.postid and p.owneruserid = u.id and pl.relatedpostid = p.id 
    and c.score = 0 and pl.creationdate >= 1299485124 and pl.linktypeid = 1 
    and p.answercount <= 4 and p.answercount >= 0 and p.creationdate <= 1410508579 
    and u.creationdate <= 1405166887 and u.creationdate >= 1280206710 
    and u.views <= 160 and v.bountyamount <= 100 and v.creationdate <= 1410364800 
    and v.creationdate >= 1233590400;
    """

    # Create QueryFeature instance
    qf = QueryFeature(query)

    # Get table information
    print("Tables:")
    for alias, info in qf.tables.items():
        print(f"Table: {info.name} (Alias: {alias})")
        print(f"Predicates: {info.predicates}")

    # Get join conditions
    print("\nJoin Conditions:")
    for join in qf.join_conditions:
        print(f"{join.table1}.{join.column1} {join.operator} {join.table2}.{join.column2}")

    # Get joins between specific tables
    tables_of_interest = ['p', 'c', 'u']
    relevant_joins = qf.get_table_joins(tables_of_interest)
    print(f"\nJoins between {tables_of_interest}:")
    for join in relevant_joins:
        print(f"{join.table1}.{join.column1} {join.operator} {join.table2}.{join.column2}")

    table_embedder, column_embedder, column_stats = initialize_database_embeddings(config.save_path)

    # Create encoder
    encoder = SubQueryEncoder(table_embedder, column_embedder, column_stats, qf)

    # Encode subquery
    table_aliases = ['v', 'p', 'u', 'c']  # Example tables involved in subquery
    predicates_matrix, predicates_list, join_mask = encoder.encode_subquery(table_aliases)

    table_matrix, join_matrix, predicates_matrix = encoder.encode_subquery_for_mscn(table_aliases)
    print(table_matrix.shape)
    print(join_matrix.shape)
    print(predicates_matrix.shape)

    # print(predicates_matrix)
    # print(predicates_list)
    print(join_mask)


if __name__ == "__main__":
    main()