import sys
import train_distillation

if __name__ == "__main__":
    # Replace with the actual path to your Wan 2.2 weights
    teacher_path = "<PATH_TO_WAN2_2_WEIGHTS>"

    # Arguments
    args = {
        "--teacher_path": teacher_path,
        "--student_config": "config/student_config.json",
        "--data_path": "data/static_prompts.txt",
        "--output_dir": "./outputs/wan_t2i",
        "--batch_size": 2,
        "--num_steps": 500,
        "--lr": 1e-5
    }

    # Convert dict to list for argparse
    sys.argv = ['main.py'] + [f"--{k}" if '=' not in v else f"--{k}={v}" for k, v in args.items()]

    train_distillation.main()