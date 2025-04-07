import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Start the card server with optional schema save/load path')
    parser.add_argument('--save_path', type=str, help='Path to save/load schema information', default='./init_model')
    parser.add_argument('--model_save_dir', type=str, help='Path to save/load model', default='./init_model/')
    parser.add_argument('--batch_size', type=int, help='Batch size', default=32)
    parser.add_argument('--num_layers', type=int, help='Number of layers', default=10)
    parser.add_argument('--timeout', type=int, help='Timeout', default=300)
    parser.add_argument('--dycard_port', type=int, help='Dycard port', default=7655)
    parser.add_argument('--dbname', type=str, help='Database name', default='stats')
    parser.add_argument('--learn_rate', type=float, help='Learning rate', default=0.0005)
    parser.add_argument('--num_epochs_per_train', type=int, help='Number of epochs per train', default=1)
    parser.add_argument('--max_sample_queue_size', type=int, help='Sample queue size', default=3000)
    parser.add_argument('--max_buffer_size', type=int, help='Maximum buffer size', default=1000)
    parser.add_argument('--delta_weight', type=float, help='Delta weight', default=2)
    parser.add_argument('--baseline', action='store_true', help='Run Baseline (PostgreSql Only)', default=False)
    parser.add_argument('--test_mode', action='store_true', help='Test mode', default=False)
    parser.add_argument('--workload', type=str, help='Workload', default='stats_800.sql')
    parser.add_argument('--pg_port', type=int, help='PostgreSql port', default=4321)
    parser.add_argument('--pg_user', type=str, help='PostgreSql user', default='ecs-user')

    return parser.parse_args()

config = parse_arguments()