set cwd="C:\Users\Michael\PycharmProjects\Brawl_iris\"
cd %cwd%

:: code upload
set arch_path=scripts\vastai\code.zip
set init_script=scripts\vastai\init.sh
del %arch_path%
7z a -tzip %arch_path% %cwd%\* -xr!outputs -xr!input_artifacts -xr!.git -xr!.idea -xr!assets

for %%x in (%arch_path% %init_script%) do (
aws s3 cp %%x s3://vastai-output/input/ --profile tmg
)

:: input artifacts
set dataset_path=input_artifacts\dataset.zip

for %%x in (%dataset_path%) do (
aws s3 cp %%x s3://vastai-output/input/input_artifacts/ --profile tmg
)

:: exit /b
timeout /t 5
vastai cloud copy --src vastai-output/input/ --dst /workspace/ --instance %1 --connection 11279 --transfer "Cloud To Instance"
vastai ssh-url %1 > instance_ssh_address.txt
set /p instance_ssh_address=<instance_ssh_address.txt
del instance_ssh_address.txt
@REM ssh %instance_ssh_address% "sh init.sh"

:: vastai cloud copy --src /workspace/Brawl_iris/outputs/better_vocab/2024-05-17_20-26-01/checkpoints --dst vastai-output/outputs/tokenizer_checkpoints/better_vocab/ --instance 10881071 --connection 11279 --transfer "Instance To Cloud"