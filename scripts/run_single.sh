# extract file name without ext
filepath=$1
file=${filepath%%.*}
file=${file##*/}

logdir=./logs
# create log dir
if [ ! -d "$logdir" ]; then
    mkdir "$logdir"
fi

# get time tag
tag=`date +'_%Y-%m-%d_%H'`

# start program
log_file=$logdir/$file$tag.log
echo Log file is $log_file
nohup python main_mbpo_vae_exp.py --config-file $1 > $log_file 2>&1 &
echo Program started