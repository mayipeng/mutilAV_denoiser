nohup python -u train.py > test.log 2>&1 &

tail test.log -n 100000 -f
