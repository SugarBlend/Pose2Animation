from argparse import ArgumentParser, Namespace
from pathlib import Path

from deploy2serve.deployment.deploy import get_object
from deploy2serve.deployment.models.export import ExportConfig


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--deploy_config", default="./configs/dynamic.yml",
                        help="Path to deploy config.")
    parser.add_argument("--pose_config",
                        default="../../configs/sapiens_pose/goliath/sapiens_0.3b-210e_goliath-1024x768.py",
                        help="Path to configuration file.")
    parser.add_argument("--pose_checkpoints",
                        default="../../weights/goliath/sapiens_0.3b/sapiens_0.3b_goliath_best_goliath_AP_573.pth",
                        help="Path to PyTorch weights file.")
    return parser.parse_args()


if __name__ == "__main__":
    import deploy2serve.utils.logger

    deploy2serve.utils.logger.get_project_root = lambda: Path(__file__).parents[2]
    root = deploy2serve.utils.logger.get_project_root()

    args = parse_arguments()
    config = ExportConfig.from_file(args.deploy_config)

    exporter = get_object(config.exporter.module_path, config.exporter.class_name)(config)

    if not Path(config.torch_weights).is_absolute():
        config.torch_weights = str(root.joinpath(config.torch_weights))

    exporter.load_checkpoints(args.pose_config, args.pose_checkpoints)
    executor = get_object(config.executor.module_path, config.executor.class_name)(config)

    for backend in config.formats:
        exporter.convert(backend)
        if config.enable_visualization:
            executor.visualization(backend)
