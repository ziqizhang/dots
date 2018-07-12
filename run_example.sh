export MKL_THREADING_LAYER=GNU
#export PYTHONPATH=/home/ziqizhang/chase/python/src
export PYTHONPATH=/home/zz/Work/chase/python/src
#input=/home/zz/Work/dots/data/w/labeled_data_all.csv
input=/home/zz/Work/chase/data/ml/ml/rm/labeled_data_all.csv
output=/home/zz/Work/dots/output
emg_model=/home/zz/Work/data/GoogleNews-vectors-negative300.bin.gz
emg_dim=300
emt_model=/home/zz/Work/data/Set1_TweetDataWithoutSpam_Word.bin
emt_dim=300
eml_model=/home/zz/Work/data/glove.840B.300d.bin.gensim
eml_dim=300
targets=2

echo "cores,mem"
nproc
free -m

SETTINGS=(
"$input $output False lstm=100-True|gmaxpooling1d|dense=$targets-softmax|ggl $emg_model" 
"$input $output False lstm=100-True|gmaxpooling1d|dense=$targets-softmax|glv $eml_model" 
"$input $output False lstm=100-True|gmaxpooling1d|dense=$targets-softmax|tw $emt_model" 
"$input $output False bilstm=100-True|gmaxpooling1d|dense=$targets-softmax|ggl $emg_model" 
"$input $output False bilstm=100-True|gmaxpooling1d|dense=$targets-softmax|glv $eml_model" 
"$input $output False bilstm=100-True|gmaxpooling1d|dense=$targets-softmax|tw $emt_model" 
"$input $output False bilstm=100-True|gmaxpooling1d|dense=$targets-softmax|ggl $emg_model" 
"$input $output False bilstm=100-True|gmaxpooling1d|dense=$targets-softmax|glv $eml_model" 
"$input $output False bilstm=100-True|gmaxpooling1d|dense=$targets-softmax|tw $emt_model" 
"$input $output False cnn[2,3,4](conv1d=100)|maxpooling1d=4|flatten|dense=$targets-softmax|ggl $emg_model"
"$input $output False cnn[2,3,4](conv1d=100)|maxpooling1d=4|flatten|dense=$targets-softmax|glv $eml_model"
"$input $output False cnn[2,3,4](conv1d=100)|maxpooling1d=4|flatten|dense=$targets-softmax|tw $emt_model"
"$input $output False scnn[2,3,4](conv1d=100,maxpooling1d=4)|maxpooling1d=4|flatten|dense=$targets-softmax|ggl $emg_model"
"$input $output False scnn[2,3,4](conv1d=100,maxpooling1d=4)|maxpooling1d=4|flatten|dense=$targets-softmax|glv $eml_model"
"$input $output False scnn[2,3,4](conv1d=100,maxpooling1d=4)|maxpooling1d=4|flatten|dense=$targets-softmax|tw $emt_model"
)

IFS=""

echo ${#SETTINGS[@]}
c=0
for s in ${SETTINGS[*]}
do
    printf '\n'
    c=$[$c +1]
    echo ">>> Start the following setting at $(date): "
    echo $c
    line="\t${s}"
    echo -e $line
    python3 -m ml.classifier ${s}
    echo "<<< completed at $(date): "
done



