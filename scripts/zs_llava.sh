DATA="data"  # Assuming this is a fixed directory
TRAINER="llava_adapt"
emb="$2"
type="none"
CFG="clip_b32"
mode_="zs"
SHOTS=1
top_bottom=5
llm="llama"
normalize=False
do_sample='True'
noise="False"
cuda="$1"  # GPU device, first argument
alpha=0.75
temperature=1.0
top_k=50
top_p=0.9
exp_name=llava_debug
global_steps=2
cross_attention='False'
all_layer_diff='False'
mistakes_context='False' 
specific_layer_diff='True'
mean_emb='True'
emavg='True'
cross_attention_middle_layer='False'
diff_layer=16
ema_alpha=1.0 # Exponential moving average alpha	
shift 2  # Remove the processed GPU argument
for data in "$@"; do
    CUDA_VISIBLE_DEVICES=$cuda python main_llava_zs.py \
        --root "${DATA}" \
        --trainer "${TRAINER}" \
        --dataset-config-file "configs/datasets/${data}.yaml" \
        --config-file "configs/trainers/adapt/${CFG}.yaml" \
        --output-dir "output/${TRAINER}/${CFG}/${data}/alpha_${alpha}" \
        --txt_epochs 100 \
        --lr 0.001 \
        --txt_cls 2 \
        --zero_shot \
        --text_emb "${emb}" \
        --corruption \
        --type "${type}" \
        --mode "${mode_}" \
        --num_shots "${SHOTS}" \
        --global_steps "${global_steps}" \
        --top_bottom "${top_bottom}" \
        --llm "${llm}" \
        --alpha "${alpha}" \
        --normalize "${normalize}" \
        --exp_name "${exp_name}" \
        --noise "${noise}" \
        --temperature "${temperature}" \
        --top_k "${top_k}" \
        --top_p "${top_p}" \
        --do_sample "${do_sample}" \
        --all_layer_diff "${all_layer_diff}" \
        --cross_attention "${cross_attention}" \
        --mistakes_context "${mistakes_context}" \
        --specific_layer_diff "${specific_layer_diff}" \
        --diff_layer "${diff_layer}" \
        --emavg "${emavg}" \
        --ema_alpha "${ema_alpha}" \
        --mean_emb "${mean_emb}" \
        --cross_attention_middle_layer "${cross_attention_middle_layer}" \
        --num_prompt 10
done