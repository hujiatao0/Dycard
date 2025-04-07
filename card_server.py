import socket
import time  
import json
import torch
from config import config   
from utils import get_model_dim, initialize_database_embeddings, extract_training_samples, visualize_attention_correlations
from model_manager import ModelManager
from feature import QueryFeature, SubQueryEncoder
import signal
import os

server_port = config.dycard_port
# export DYCARD_SERVER_PORT=7654

def print_plan_tree(plan, indent=0):
    indent_str = ' ' * indent
    print(f"{indent_str}Node Type: {plan.get('node_type')}")
    print(f"{indent_str}  Tables: {', '.join(plan.get('tables', []))}")
    print(f"{indent_str}  Estimated Rows: {plan.get('estimated_rows')}")
    print(f"{indent_str}  Actual Rows: {plan.get('actual_rows')}")
    print(f"{indent_str}  Loops: {plan.get('loops')}")
    children = plan.get('children', [])
    for child in children:
        print_plan_tree(child, indent + 4)

def clean_json_string(json_str):
    json_str = json_str.replace(',}', '}')
    json_str = json_str.replace(',]', ']')
    return json_str

def save_model_on_exit(model_manager):
    """Signal handler function to save model before program exits"""
    print("Saving model before exit...")
    model_manager.save_infer_model()
    print("Model saved successfully.")
    exit(0)

def main():
    
    try:

        # Initialize database embeddings with optional save path
        table_embedder, column_embedder, column_stats = initialize_database_embeddings(config.save_path)
        encoder = SubQueryEncoder(table_embedder, column_embedder, column_stats, None)
        model_dim = get_model_dim(len(table_embedder.table_to_idx), len(column_embedder.column_to_idx))
        print(f"Model dimension: {model_dim}")
        model_manager = ModelManager(model_dim, encoder)
        model_manager.start_training_process()
        # Register signal handlers
        signal.signal(signal.SIGINT, lambda signum, frame: save_model_on_exit(model_manager))
        signal.signal(signal.SIGTERM, lambda signum, frame: save_model_on_exit(model_manager))

        # Initialize socket server
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(('localhost', server_port))
        server_socket.listen(1)
        print(f'Server listening on port {server_port}...')
        
        while True:  
            client_socket, client_address = server_socket.accept()
            print(f'Received connection from {client_address}')

            try:
                while True: 
                    try:
                        data = ""
                        while True:
                            chunk = client_socket.recv(4096).decode('utf-8')
                            if not chunk:  
                                raise ConnectionResetError("Client disconnected")
                            data += chunk
                            if '\n' in chunk: 
                                break
                                
                        message = data.strip()
                        if message:
                            # print(f'Received message: {message}')
                            
                            if message.startswith('subplan:'):
                                # print(message[len('subplan:'):].strip())
                                prediction, attentions, all_predicates = model_manager.infer(message[len('subplan:'):].strip())
                                # print(f"all_predicates: {all_predicates}")
                                # visualize_attention_correlations(attentions, all_predicates, save_prefix="")
                                # print(f"Prediction: {prediction}")
                                client_socket.send(str(prediction).encode('utf-8'))
                            elif message.startswith('plan:'):
                                plan_json = message[len('plan:'):].strip()
                                try:
                                    cleaned_json = clean_json_string(plan_json)
                                    plan_data = json.loads(cleaned_json)
                                    print(f"Plan data: {plan_data}")    
                                    # print_plan_tree(plan_data)

                                    training_samples = extract_training_samples(plan_data)
                                    for tables, est_rows, act_rows in training_samples:
                                        # if len(tables) <= 1:
                                        #     continue
                                        model_manager.add_training_sample((tables, est_rows, act_rows))
                                        print(f"Added training sample: tables={tables}, estimated_rows={est_rows}, actual_rows={act_rows}")

                                except json.JSONDecodeError as e:
                                    print(f"Error decoding JSON: {str(e)}")
                                    print(f"Problematic JSON: {plan_json}")
                                # client_socket.send(str(num).encode('utf-8'))
                                # num += 1
                            elif message.startswith('query:'):
                                message = message.lower()
                                if 'explain analyze' in message:
                                    qf = QueryFeature(message[len('query:'):].strip())
                                    model_manager.update_query_feature(qf)
                                client_socket.send("done".encode('utf-8'))
                            
                    except ConnectionResetError:
                        print("Client disconnected")
                        break
                    except Exception as e:
                        print(f'Error in message handling: {str(e)}')

            finally:
                print("Closing current client connection...")
                client_socket.close()
                
    except Exception as e:
        print(f"Server initialization failed: {str(e)}")
        return
    
    finally:
        if 'server_socket' in locals():
            server_socket.close()

if __name__ == "__main__":
    main()