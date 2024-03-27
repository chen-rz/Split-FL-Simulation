#!/bin/bash
rm -rf /home/crz/logs
mkdir /home/crz/logs
cd FL_training
for i in {1..5}
do
    touch /home/crz/logs/Client-$i.log
    nohup python -u FedAdapt_clientrun.py --offload t --hostname Client-$i > /home/crz/logs/Client-$i.log 2>&1 &
done