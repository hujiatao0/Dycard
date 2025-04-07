import threading
import psycopg2
import time
from pathlib import Path
import logging
from datetime import datetime
import re
from config import config
import os
# Configure logging

def calculate_q_error(est_card: float, true_card: float) -> float:
    """Calculate q-error for a single node"""
    # Handle zero values
    if est_card == 0 and true_card == 0:
        return 1.0
    if est_card == 0:
        est_card = 1.0
    if true_card == 0:
        true_card = 1.0
    
    return max(est_card / true_card, true_card / est_card)

class QueryExecutor:
    def __init__(self, timeout=60, parallel_mode=False, base_mode=True):
        self.timeout = timeout
        self.parallel_mode = parallel_mode
        self.base_mode = base_mode

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)

        print(f"config: {config}")
        
        if self.base_mode:
            if not self.parallel_mode:
                self.results_file = f"./results/{config.dbname}_train_baseline.txt"
            else:
                self.results_file = f"./results/{config.dbname}_normal_baseline.txt"
        else:
            if not self.parallel_mode:
                self.results_file = f'./results/{config.dbname}_loss{config.delta_weight}_layer{config.num_layers}_learnrate{config.learn_rate}_{config.num_epochs_per_train}per_train.txt'
            else:
                self.results_file = f'./results/{config.dbname}_loss{config.delta_weight}_layer{config.num_layers}_learnrate{config.learn_rate}_{config.num_epochs_per_train}per_test.txt'
        self.logger.info(f"Results will be written to: {self.results_file}")
    
    def connect_db(self):
        """Connect to database"""
        return psycopg2.connect(
            user=config.pg_user,
            host="localhost",
            port=config.pg_port,
            database=config.dbname
        )
    
    def load_queries(self):
        """Load queries"""
        queries = []
        query_path = Path(f"workload/{config.workload}")
        with open(query_path, 'r') as f:
            queries = f.readlines()
        return queries
    
    def execute_query(self, cur, query, query_id, timeout_seconds=None):
        """Execute a single query and get execution time"""
        if timeout_seconds is None:
            timeout_seconds = self.timeout
        
        self.logger.info(f"Starting execution of query {query_id}")
        start_time = time.time()
        
        result = {
            'query_id': query_id,
            'status': None,
            'planning_time': None,
            'execution_time': None,
            'q_error': None
        }
        
        def query_executor():
            self.logger.info(f"SET static_mode = {self.parallel_mode}")
            try:
                # Set basic parameters
                if self.base_mode and not self.parallel_mode:
                    cur.execute("SET max_parallel_workers_per_gather = 0;")
                    cur.execute("SET max_parallel_workers = 0;")
                elif not self.base_mode:
                    cur.execute("SET enable_dycard = true;")
                    cur.execute(f"SET dycard.port = {config.dycard_port};")
                    if self.parallel_mode:
                        cur.execute("SET static_mode = true;")

                explain_query = f"EXPLAIN ANALYZE {query}"
                cur.execute(explain_query)
                results = cur.fetchall()
                
                max_q_error = 1.0  # Initialize max q-error
                
                for row in results:
                    line = row[0]
                    # Skip index-related and materialize operators
                    if "Index" in line or "Materialize" in line:
                        continue
                    if "Planning Time" in line:
                        result['planning_time'] = float(line.split("Planning Time: ")[1].split(" ms")[0])
                    elif "Execution Time" in line:
                        result['execution_time'] = float(line.split("Execution Time: ")[1].split(" ms")[0])
                    else:
                        est_match = re.search(r"rows=(\d+)", line)
                        act_match = re.search(r"actual time=.*?rows=(\d+)", line)
                        if est_match and act_match:
                            est = float(est_match.group(1))
                            act = float(act_match.group(1))
                            q_err_node = calculate_q_error(est, act)
                            if q_err_node > max_q_error:
                                max_q_error = q_err_node
                
                result['q_error'] = max_q_error
                result['status'] = 'completed'
            except Exception as e:
                result['status'] = 'error'
                result['error'] = str(e)
        
        query_thread = threading.Thread(target=query_executor)
        query_thread.daemon = True
        query_thread.start()
        query_thread.join(timeout=timeout_seconds)
        
        if query_thread.is_alive():
            self.logger.warning(f"Query {query_id} timed out (timeout: {timeout_seconds}s), cancelling...")
            conn = cur.connection
            pid = conn.get_backend_pid()
            cancel_conn = self.connect_db()
            try:
                cancel_cur = cancel_conn.cursor()
                cancel_cur.execute(f"SELECT pg_terminate_backend({pid});")
                cancel_conn.commit()
            finally:
                cancel_conn.close()
            result['status'] = 'timeout'
        
        total_time = time.time() - start_time
        if result['status'] == 'completed':
            self.logger.info(f"Query {query_id} completed, total time: {total_time:.2f}s")
        
        return result
    
    def evaluate_batch(self, eval_results):
        """Evaluation function (placeholder)"""
        pass
    
    def run(self):
        """Run query execution process"""
        queries = self.load_queries()
        self.logger.info(f"Starting execution, total {len(queries)} queries")
        
        for query_id, query in enumerate(queries):
            conn = self.connect_db()
            cur = conn.cursor()
            
            try:
                result = self.execute_query(cur, query, query_id)
                
                # Record results
                os.makedirs(os.path.dirname(self.results_file), exist_ok=True)
                with open(self.results_file, 'a') as f:
                    if result['status'] == 'completed':
                        f.write(f"Query {query_id}: Planning={result['planning_time']}ms, "
                               f"Execution={result['execution_time']}ms, "
                               f"Q-error={result['q_error']}\n")
                    elif result['status'] == 'timeout':
                        f.write(f"Query {query_id}: timeout\n")
                    else:
                        f.write(f"Query {query_id}: error\n")
            
            except Exception as e:
                self.logger.error(f"Error executing query {query_id}: {str(e)}")
                with open(self.results_file, 'a') as f:
                    f.write(f"Query {query_id}: error\n")
            
            finally:
                cur.close()
                conn.close()
        
        self.logger.info("All queries completed")

if __name__ == "__main__":
    executor = QueryExecutor(timeout=config.timeout, parallel_mode=config.test_mode, base_mode=config.baseline)
    executor.run()
