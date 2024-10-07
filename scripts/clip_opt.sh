DATA="data"  # Assuming this is a fixed directory
TRAINER="clip_adapt"
emb="s_temp"
type="none"
CFG="clip_b32"
mode_="opt"
SHOTS=1
top_bottom=5
llm="llama"
normalize=False
do_sample='True'
noise="False"
cuda="$1"  # GPU device, first argument
alpha=0.0
temperature=1.0
top_k=50
top_p=0.9
exp_name="clip_adapt"
global_steps=100
cross_attention='False'
all_layer_diff='False'
mistakes_context='False' 
specific_layer_diff='True'
mean_emb='True'
emavg='False'
cross_attention_middle_layer='False'
diff_layer=17
ema_alpha=1.0 # Exponential moving average alpha	
shift 1  # Remove the processed GPU argument
alphas=(1.0 0.75 0.5 0.25)
num_prompt=1
# Iterate over alpha values
for alpha in "${alphas[@]}"; do
    for data in "$@"; do
        CUDA_VISIBLE_DEVICES="${cuda}" python main.py \
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
            --num_prompt "${num_prompt}"
    done
done