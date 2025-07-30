uv venv
uv sync --all-extras
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v2.1.0
uv run python setup.py bdist_wheel
uv pip install ./dist/mmcv-2.1.0-cp310-cp310-win_amd64.whl
cd ..
rmdir /s /q mmcv

@REM For using export module
git clone https://github.com/SugarBlend/-DeployAndServe.git
cd "-DeployAndServe"
poetry build
uv pip install dist/deploy2serve-0.1.0-py3-none-any.whl --no-deps
uv pip install pydantic colorlog onnx onnxslim onnxruntime-gpu tensorrt==10.10.0.31 numpy==1.23.5
cd ..
rmdir /s /q "-DeployAndServe"

uv run pre-commit install
