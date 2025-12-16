# cs4820-project
Reinforcement learning minesweeper

To install torch that is compatible with a 5080:
python -m venv venv_3.12.6
venv_3.12.6\Scripts\Activate.ps1
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements.txt
python .\main_tester_file.py