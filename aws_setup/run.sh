cd ~
eval "$(conda shell.bash hook)"
conda activate brawl_stars_cloud
cd Brawl_iris
python src/train_cloud.py || :
cd ~
aws s3 cp checkpoints s3://brawl-stars-iris/"$1"/checkpoints --recursive --exclude "dataset/*"
rm -r checkpoints
rm -r Brawl_iris
echo "Removed repo dir and chkpt"
ls ~