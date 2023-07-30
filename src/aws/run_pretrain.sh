cd ~
eval "$(conda shell.bash hook)"
conda activate brawl_stars_cloud
cd Brawl_iris
python -c "from src.pretraining import init_dataset; init_dataset()"
python src/train_cloud.py ++run_prefix="$1" || :
cd ~
rm -r checkpoints
rm -r Brawl_iris
echo "Removed repo dir and chkpt"
