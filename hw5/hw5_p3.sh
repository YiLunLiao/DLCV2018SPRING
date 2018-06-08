wget https://www.dropbox.com/s/jovy1nxbybel81t/seq2seq_cnn.pt?dl=1 -O seq2seq_cnn.pt
wget https://www.dropbox.com/s/9q3oxrfozle55af/seq2seq_rnn.pt?dl=1 -O seq2seq_rnn.pt
python3 hw5_seq2seq_test.py $1 $2
