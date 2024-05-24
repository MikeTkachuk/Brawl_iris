# shellcheck disable=SC2164
apt-get update && apt-get install unzip ffmpeg libsm6 libxext6  -y
unzip code.zip -d Brawl_iris
mv input_artifacts Brawl_iris/input_artifacts -v -f
cd Brawl_iris
conda env create -f scripts/vastai/environment.yaml
eval "$(conda shell.bash hook)"
conda init bash
conda activate brawl_stars
wandb login $WANDB_API_KEY

python src/train_components/train_world_model.py