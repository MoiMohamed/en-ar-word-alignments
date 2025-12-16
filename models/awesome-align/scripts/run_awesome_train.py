#!/usr/bin/env python3
"""
Fine-tune awesome-align model on parallel data.
"""

import json
import os
import subprocess
import sys

def convert_json_to_awesome_format(json_file, output_file, max_sentences=None):
    """Convert JSON file to awesome-align input format."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Limit number of sentences if specified
    if max_sentences is not None and max_sentences > 0:
        data = data[:max_sentences]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in data:
            ar = entry['ar']
            en = entry['en']
            f.write(f'{ar} ||| {en}\n')
    
    return len(data)

def main():
    import argparse
    
    # Get script directory for relative path resolution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, '../../..'))
    awesome_align_dir = os.path.normpath(os.path.join(script_dir, '..'))
    
    # Default paths (release structure)
    default_train = os.path.normpath(os.path.join(project_root, 'data/BASELINE_A/train.json'))
    default_dev = os.path.normpath(os.path.join(project_root, 'data/BASELINE_A/dev.json'))
    default_output = os.path.normpath(os.path.join(awesome_align_dir, 'fine_tuned_custom'))
    default_cache = os.path.normpath(os.path.join(awesome_align_dir, 'cache'))
    
    parser = argparse.ArgumentParser(description='Fine-tune awesome-align model on parallel data')
    parser.add_argument('--train', type=str, default=default_train,
                       help='Training JSON file (default: ../../data/BASELINE_A/train.json)')
    parser.add_argument('--dev', type=str, default=default_dev,
                       help='Development JSON file (default: ../../data/BASELINE_A/dev.json)')
    parser.add_argument('--model', type=str, default='bert-base-multilingual-cased',
                       help='Base model name or path (default: bert-base-multilingual-cased)')
    parser.add_argument('--output-dir', type=str, default=default_output,
                       help='Output directory for fine-tuned model (default: ../fine_tuned_custom)')
    parser.add_argument('--cache-dir', type=str, default=default_cache,
                       help=f'Directory to cache models (default: {default_cache})')
    parser.add_argument('--max-train-sentences', type=int, default=None,
                       help='Maximum number of training sentences to use (default: all)')
    parser.add_argument('--max-dev-sentences', type=int, default=None,
                       help='Maximum number of dev sentences to use (default: all)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID to use (default: 0, set to -1 for CPU)')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Per GPU batch size (default: 2)')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4,
                       help='Gradient accumulation steps (default: 4)')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                       help='Learning rate (default: 2e-5)')
    parser.add_argument('--num-epochs', type=int, default=1,
                       help='Number of training epochs (default: 1)')
    parser.add_argument('--max-steps', type=int, default=20000,
                       help='Maximum training steps (default: 20000)')
    parser.add_argument('--save-steps', type=int, default=4000,
                       help='Save checkpoint every N steps (default: 4000)')
    parser.add_argument('--train-tlm', action='store_true',
                       help='Use translation language modeling objective')
    parser.add_argument('--train-mlm', action='store_true',
                       help='Use masked language modeling objective')
    parser.add_argument('--train-tlm-full', action='store_true',
                       help='Use full translation language modeling')
    parser.add_argument('--train-so', action='store_true',
                       help='Use self-training objective (recommended)')
    parser.add_argument('--train-psi', action='store_true',
                       help='Use parallel sentence identification objective')
    parser.add_argument('--train-co', action='store_true',
                       help='Use contrastive objective (may lower precision)')
    parser.add_argument('--overwrite-output-dir', action='store_true',
                       help='Overwrite output directory if it exists')
    parser.add_argument('--keep-temp', action='store_true',
                       help='Keep temporary input files after processing')
    
    args = parser.parse_args()
    
    # Resolve relative paths to absolute paths
    if not os.path.isabs(args.train):
        args.train = os.path.abspath(os.path.join(os.getcwd(), args.train))
    else:
        args.train = os.path.normpath(args.train)
    
    if args.dev:
        if not os.path.isabs(args.dev):
            args.dev = os.path.abspath(os.path.join(os.getcwd(), args.dev))
        else:
            args.dev = os.path.normpath(args.dev)
    
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.abspath(os.path.join(os.getcwd(), args.output_dir))
    else:
        args.output_dir = os.path.normpath(args.output_dir)
    
    if not os.path.isabs(args.cache_dir):
        args.cache_dir = os.path.abspath(os.path.join(os.getcwd(), args.cache_dir))
    else:
        args.cache_dir = os.path.normpath(args.cache_dir)
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    print('=' * 80)
    print('Fine-tuning awesome-align model')
    print('=' * 80)
    
    # Verify input files exist
    if not os.path.exists(args.train):
        print(f'\nError: Training file not found: {args.train}')
        sys.exit(1)
    
    if args.dev and not os.path.exists(args.dev):
        print(f'\nError: Development file not found: {args.dev}')
        sys.exit(1)
    
    # Convert JSON to awesome-align format
    print(f'\nConverting training data...')
    train_input = os.path.join(args.output_dir, 'train_input.src-tgt')
    num_train = convert_json_to_awesome_format(args.train, train_input, args.max_train_sentences)
    print(f'  Converted {num_train} training sentence pairs')
    print(f'  Training file: {train_input}')
    
    dev_input = None
    if args.dev:
        print(f'\nConverting development data...')
        dev_input = os.path.join(args.output_dir, 'dev_input.src-tgt')
        num_dev = convert_json_to_awesome_format(args.dev, dev_input, args.max_dev_sentences)
        print(f'  Converted {num_dev} development sentence pairs')
        print(f'  Development file: {dev_input}')
    
    # Set up environment for GPU and OpenBLAS
    env = os.environ.copy()
    if args.gpu >= 0:
        env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    else:
        env['CUDA_VISIBLE_DEVICES'] = ''
    
    # Limit OpenBLAS threads to avoid resource limit issues
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['OMP_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    
    # Build awesome-train command
    gpu_msg = f'GPU {args.gpu}' if args.gpu >= 0 else 'CPU'
    print(f'\nFine-tuning model: {args.model} on {gpu_msg}...')
    print(f'  Output directory: {args.output_dir}')
    print(f'  Cache directory: {args.cache_dir}')
    print(f'  Training steps: max {args.max_steps}, save every {args.save_steps}')
    print('  Processing...')
    
    cmd = [
        'awesome-train',
        '--output_dir', args.output_dir,
        '--model_name_or_path', args.model,
        '--extraction', 'softmax',
        '--do_train',
        '--train_data_file', train_input,
        '--per_gpu_train_batch_size', str(args.batch_size),
        '--gradient_accumulation_steps', str(args.gradient_accumulation_steps),
        '--learning_rate', str(args.learning_rate),
        '--num_train_epochs', str(args.num_epochs),
        '--max_steps', str(args.max_steps),
        '--save_steps', str(args.save_steps),
        '--cache_dir', args.cache_dir
    ]
    
    # Add overwrite flag if specified
    if args.overwrite_output_dir:
        cmd.append('--overwrite_output_dir')
    
    # Add training objectives
    if args.train_tlm:
        cmd.append('--train_tlm')
    if args.train_mlm:
        cmd.append('--train_mlm')
    if args.train_tlm_full:
        cmd.append('--train_tlm_full')
    if args.train_so:
        cmd.append('--train_so')
    if args.train_psi:
        cmd.append('--train_psi')
    if args.train_co:
        cmd.append('--train_co')
    
    # Add evaluation if dev file is provided
    if args.dev:
        cmd.extend(['--do_eval', '--eval_data_file', dev_input])
    
    # If no training objectives specified, use default (TLM + SO)
    if not any([args.train_tlm, args.train_mlm, args.train_tlm_full, 
                args.train_so, args.train_psi, args.train_co]):
        cmd.append('--train_tlm')
        cmd.append('--train_so')
        print('  Using default training objectives: --train_tlm --train_so')
    
    try:
        # Run with real-time output to show progress
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                   text=True, env=env, bufsize=1, universal_newlines=True)
        
        # Show output in real-time
        for line in process.stdout:
            print(f'  {line.rstrip()}')
        
        process.wait()
        
        if process.returncode != 0:
            print(f'  Error: awesome-train exited with code {process.returncode}')
            sys.exit(1)
        
        print('\n  Fine-tuning completed successfully!')
    except subprocess.CalledProcessError as e:
        print(f'  Error running awesome-train: {e}')
        if hasattr(e, 'stderr') and e.stderr:
            print(f'  stderr: {e.stderr}')
        sys.exit(1)
    finally:
        # Clean up temp files unless --keep-temp is set
        if not args.keep_temp:
            if os.path.exists(train_input):
                os.remove(train_input)
                print(f'  Cleaned up temporary file: {train_input}')
            if os.path.exists(dev_input):
                os.remove(dev_input)
                print(f'  Cleaned up temporary file: {dev_input}')
        elif args.keep_temp:
            print(f'  Kept temporary files: {train_input}, {dev_input}')
    
    print('\n' + '=' * 80)
    print(f'Fine-tuned model saved to: {args.output_dir}')
    print('=' * 80)

if __name__ == '__main__':
    main()

