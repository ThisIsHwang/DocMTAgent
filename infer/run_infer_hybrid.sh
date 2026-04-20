lang=en-ko_KR
src_lang=${lang%%-*}
tgt_lang=${lang##*-}

workflow=auto
model_platform=openai
model_name=gpt-4o-mini

summary_step=20
long_window=20
top_k=2
context_window=3
chunk_size_sentences=1

quality_threshold=26
max_iterations=6

pyfile=infer_hybrid.py

out_path=results_hybrid

src=/path/to/src/file
ref=/path/to/ref/file

src_summary_prompt=prompts/${lang}/src_summary_prompt.txt
tgt_summary_prompt=prompts/${lang}/tgt_summary_prompt.txt
src_merge_prompt=prompts/${lang}/src_merge_prompt.txt
tgt_merge_prompt=prompts/${lang}/tgt_merge_prompt.txt
history_prompt=prompts/${lang}/history_prompt.txt
retrieve_prompt=prompts/retrieve_prompt.txt

if [ ! -d $out_path ]; then
    mkdir -p $out_path
fi

output=$out_path/${workflow}_${model_name}.json

cmd=(
    python -u $pyfile
    --language ${lang}
    --src $src
    --output $output
    --src_summary_prompt $src_summary_prompt
    --tgt_summary_prompt $tgt_summary_prompt
    --src_merge_prompt $src_merge_prompt
    --tgt_merge_prompt $tgt_merge_prompt
    --retrieve_prompt $retrieve_prompt
    --history_prompt $history_prompt
    --summary_step $summary_step
    --long_window $long_window
    --top_k $top_k
    --chunk_size_sentences $chunk_size_sentences
    --settings summary long context history
    --context_window $context_window
    --retriever agent
    --workflow $workflow
    --model_platform $model_platform
    --model_name $model_name
    --quality_threshold $quality_threshold
    --max_iterations $max_iterations
)

if [[ $ref != '' ]]; then
    cmd+=(--ref $ref)
fi

"${cmd[@]}"

python post_process.py $output
