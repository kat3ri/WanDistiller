import sys
import unittest
from unittest.mock import MagicMock, patch, mock_open
import tempfile
import json

# -------------------------------------------------------------------------
# UPDATE: Change 'your_module' to 'train_distillation'
# -------------------------------------------------------------------------
import train_distillation


class TestFullPipeline(unittest.TestCase):

    def setUp(self):
        # Create a temporary config file mimicking student_config.json
        self.test_config = {
            "model_type": "WanLiteStudent",
            "hidden_size": 1024,
            "depth": 16,
            "num_heads": 16,
            "num_channels": 4,
            "image_size": 1024,
            "patch_size": 16,
            "text_max_length": 77,
            "text_encoder_output_dim": 4096,
            "projection_factor": 1.0
        }

        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(self.test_config, self.temp_file)
        self.temp_file.close()

        # Setup args matching main.py logic
        self.mock_args = {
            "teacher_path": "test_teacher_weights",  # Placeholder, usually mocks in test
            "student_config": self.temp_file.name,  # Points to our temp file
            "data_path": "data/static_prompts.txt",  # Matches your file
            "output_dir": "test_output",  # Matches your file
            "batch_size": 2,
            "num_steps": 50,
            "lr": 1e-5
        }

    def tearDown(self):
        import os
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    # -------------------------------------------------------------------------
    # Unit Tests
    # -------------------------------------------------------------------------

    def test_load_config_valid_json(self):
        config = train_distillation.load_config(self.temp_file.name)

        # Verify keys exist
        self.assertIn("num_channels", config)
        self.assertEqual(config["num_channels"], 4)
        self.assertEqual(config["image_size"], 1024)

    # -------------------------------------------------------------------------
    # Integration Tests for main
    # -------------------------------------------------------------------------

    @patch('torch.device')
    @patch('torch.randn')
    # UPDATE: Patch the specific class from train_distillation
    @patch('train_distillation.WanLiteStudent')
    @patch('train_distillation.DiffusionPipeline')  # Mock the teacher loading
    @patch('train_distillation.StaticPromptsDataset')
    # UPDATE: Patch load_and_project_weights based on your projection_mapper.py
    @patch('train_distillation.load_and_project_weights')
    def test_main_integration(self, mock_proj, mock_dataset, mock_diffusion, mock_student, mock_randn, mock_device):
        """
        This test ensures that:
        1. load_config is called.
        2. WanLiteStudent is initialized.
        3. load_and_project_weights is called.
        4. StaticPromptsDataset is used.
        """

        # Mock return values
        mock_device.return_value = "cpu"
        mock_randn.return_value = "mock_latents_tensor"

        # Mock Teacher Pipeline
        mock_diffusion.return_value = MagicMock(transformer=MagicMock())

        # Mock Dataset
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.__len__ = MagicMock(return_value=1)
        mock_batch = {'text_embeddings': 'mock_text'}
        mock_dataset_instance.__iter__ = MagicMock(return_value=[mock_batch])

        # Mock Student
        mock_student_instance = MagicMock()
        mock_student_instance.train = MagicMock()
        mock_student_instance.save_pretrained = MagicMock()
        mock_student.return_value = mock_student_instance

        # Setup sys.argv to match main.py style
        sys.argv = [
            "test_distillation.py",
            "--teacher_path", self.mock_args["teacher_path"],
            "--student_config", self.mock_args["student_config"],
            "--data_path", self.mock_args["data_path"],
            "--output_dir", self.mock_args["output_dir"],
            "--batch_size", self.mock_args["batch_size"],
            "--lr", str(self.mock_args["lr"])
        ]

        # Run the main function from train_distillation
        train_distillation.main()

        # Assertions
        # 1. Verify WanLiteStudent was initialized with config
        mock_student.assert_called_once_with(self.test_config)

        # 2. Verify load_and_project_weights was called
        mock_proj.assert_called_once()

        # 3. Verify save_pretrained was called with output_dir
        mock_student_instance.save_pretrained.assert_called_once_with(self.mock_args["output_dir"])


if __name__ == '__main__':
    unittest.main()

