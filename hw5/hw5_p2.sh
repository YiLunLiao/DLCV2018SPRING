wget https://www.dropbox.com/s/5wgp300vcpuy8ka/crvr_cnn.pt?dl=1 -O crvr_cnn.pt
wget https://www.dropbox.com/s/t2u8hzqjtc0j8v7/crvr_rnn.pt?dl=1 -O crvr_rnn.pt
python3 hw5_CRVarLen_test_v2.py $1 $2 $3
