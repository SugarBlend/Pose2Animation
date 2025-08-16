uv venv
uv sync --all-extras
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v2.1.0
uv run python setup.py bdist_wheel
uv pip install ./dist/mmcv-2.1.0-cp310-cp310-win_amd64.whl
cd ..

@REM For using export module
git clone https://github.com/SugarBlend/-DeployAndServe.git
cd "-DeployAndServe"
poetry build
uv pip install dist/deploy2serve-0.1.0-py3-none-any.whl --no-deps
uv pip install pydantic colorlog onnx onnxslim onnxruntime-gpu==1.19.2 tensorrt==10.10.0.31 gdown roboflow zarr h5py
uv pip uninstall opencv-python-headless
uv pip install --force-reinstall opencv-python==4.8.1.78 numpy==1.26.4

cd ..
rmdir /s /q "-DeployAndServe"

uv run pre-commit install
uv pip install -e src/triangulation/cython -v
