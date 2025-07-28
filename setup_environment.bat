uv venv
uv sync --all-extras
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v2.1.0
uv run python setup.py bdist_wheel
uv pip install ./dist/mmcv-2.1.0-cp310-cp310-win_amd64.whl
cd ..
rmdir /s /q mmcv
uv run pre-commit install
