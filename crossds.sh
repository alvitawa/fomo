
for dataset in imagenet fgvc_aircraft caltech101 oxford_pets stanford_cars food101 sun397 ucf101 dtd oxford_flowers eurosat; do
  output_dir="/home/ataboadawarmer/data/fomo/output/crossdsWW/test_new/${dataset}/shots_16/CoPrompt/seed1"
  mkdir -p "$output_dir"
  bash scripts/xd_test.sh $dataset 1 crossdsWW 32 > "${output_dir}/output.log" 2>&1
  cat "${output_dir}/output.log"
done