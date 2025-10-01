import os
import random
from pathlib import Path
import re

class TPCDSQueryLoader:
    def __init__(self, tpcds_dir="workload/tpcds"):
        self.tpcds_dir = Path(tpcds_dir)
        self.query_files = []
        self.is_test_dir = "test" in str(tpcds_dir).lower()
        self.load_query_files()

    def load_query_files(self):
        """Load all TPC-DS query files"""
        if not self.tpcds_dir.exists():
            raise FileNotFoundError(f"TPC-DS directory not found: {self.tpcds_dir}")

        if self.is_test_dir:
            # For test directory, load test_query*.sql files (no underscore)
            query_pattern = re.compile(r'test_query(\d+)\.sql')
            for file_path in self.tpcds_dir.glob("test_query*.sql"):
                match = query_pattern.match(file_path.name)
                if match:
                    query_num = int(match.group(1))
                    self.query_files.append((query_num, file_path))
        else:
            # For training directory, load query*.sql files
            query_pattern = re.compile(r'query(\d+)\.sql')
            for file_path in self.tpcds_dir.glob("query*.sql"):
                match = query_pattern.match(file_path.name)
                if match:
                    query_num = int(match.group(1))
                    self.query_files.append((query_num, file_path))

        # Sort by query number
        self.query_files.sort(key=lambda x: x[0])
        query_type = "test" if self.is_test_dir else "training"
        print(f"Loaded {len(self.query_files)} TPC-DS {query_type} queries")

    def read_query_from_file(self, file_path):
        """Read and clean a single query from file"""
        with open(file_path, 'r') as f:
            content = f.read()

        # Remove comments and empty lines
        lines = []
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('--'):
                lines.append(line)

        # Join lines and clean up
        query = ' '.join(lines)
        # Remove trailing semicolon if present
        query = query.rstrip(';').strip()

        return query

    def get_all_queries(self):
        """Get all TPC-DS queries as a list"""
        queries = []
        for query_num, file_path in self.query_files:
            try:
                query = self.read_query_from_file(file_path)
                if query:  # Only add non-empty queries
                    queries.append(query)
            except Exception as e:
                print(f"Error reading query {query_num}: {e}")

        return queries

    def get_random_query(self):
        """Get a random TPC-DS query"""
        if not self.query_files:
            return None

        query_num, file_path = random.choice(self.query_files)
        try:
            return self.read_query_from_file(file_path)
        except Exception as e:
            print(f"Error reading random query {query_num}: {e}")
            return None

    def generate_query_sequence(self, num_queries, random_distribution=True):
        """Generate a sequence of TPC-DS queries

        Args:
            num_queries: Number of queries to generate
            random_distribution: If True, randomly sample queries with replacement
                                If False, cycle through all queries in order
        """
        if not self.query_files:
            return []

        all_queries = self.get_all_queries()
        if not all_queries:
            return []

        if random_distribution:
            # Random sampling with replacement
            return [random.choice(all_queries) for _ in range(num_queries)]
        else:
            # Cycle through queries in order
            queries = []
            for i in range(num_queries):
                query_idx = i % len(all_queries)
                queries.append(all_queries[query_idx])
            return queries

if __name__ == "__main__":
    # Test the loader
    loader = TPCDSQueryLoader()

    # Test getting all queries
    all_queries = loader.get_all_queries()
    print(f"Total queries loaded: {len(all_queries)}")

    # Test getting a random query
    random_query = loader.get_random_query()
    if random_query:
        print(f"Random query preview: {random_query[:100]}...")

    # Test generating a sequence
    sequence = loader.generate_query_sequence(5, random_distribution=True)
    print(f"Generated sequence of {len(sequence)} queries")