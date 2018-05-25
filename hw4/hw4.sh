wget https://www.dropbox.com/s/n17ncnctmk153jh/gen_26.pt?dl=1 -O gen_26.pt
wget https://www.dropbox.com/s/0o8f2zpyqb5w0n0/gen_27.pt?dl=1 -O gen_27.pt
wget https://www.dropbox.com/s/wrqv149gorio9jw/vae_20.pt?dl=1 -O vae_20.pt
python3 plotLog.py --out_dir $2
python3 hw4_vae_test.py --input_dir $1 --out_dir $2
python3 hw4_vae_p4.py --out_dir $2
python3 hw4_vae_p5.py --input_dir $1 --out_dir $2
python3 hw4_DCGAN_p3.py --out_dir $2
python3 hw4_ACGAN_predictFixedVec.py --out_dir $2
