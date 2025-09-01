# srun ..
# singularity run --nv --home /projects/t5c/public/ nemo:25.02.sif (nemo:25.02.sif could be in run_last)

pip install -U "huggingface_hub[cli]"
export PATH="/projects/t5c/public/.local/bin:$PATH"
hf auth login


export MODEL=/projects/t5c/public/Llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct.nemo
export TRAIN_DS="[/projects/t5c/public/databricks-dolly-15k/training.jsonl]"
export VALID_DS="[/projects/t5c/public/databricks-dolly-15k/validation.jsonl]"
export TEST_DS="[/projects/t5c/public/databricks-dolly-15k/test.jsonl]"
export VALID_NAMES="[databricks-dolly-15k]"
export CONCAT_SAMPLING_PROBS="[1]"
export TP_SIZE=1
export PP_SIZE=1
export NEMO_LOG_LEVEL=ERROR

export TMPDIR=/projects/t5c/public/tmp
export TEMP=/projects/t5c/public/tmp
export TMP=/projects/t5c/public/tmp

unset SLURM_JOB_ID SLURM_NTASKS SLURM_NODELIST SLURM_NNODES

#full SFT - OOM
NEMO_LOG_LEVEL=ERROR \ 
nsys profile -y 360 -d 720 \ 
--trace=cuda,cudnn,cublas,osrt,nvtx \
--event-sample=system-wide \
-w true -c cudaProfilerApi -o sft_llama2_7B_4workers \
python3 /projects/t5c/public/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py \
trainer.precision=bf16 \
trainer.devices=1 \
trainer.num_nodes=1 \
trainer.val_check_interval=0.1 \
trainer.max_steps=50 \
model.restore_from_path=${MODEL} model.micro_batch_size=1 \
model.global_batch_size=128 \
model.tensor_model_parallel_size=${TP_SIZE} \
model.pipeline_model_parallel_size=${PP_SIZE} \
model.megatron_amp_O2=True \
model.sequence_parallel=False \
model.activations_checkpoint_granularity=selective \
model.activations_checkpoint_method=uniform \
model.optim.name=distributed_fused_adam \
model.optim.lr=5e-6 \
model.answer_only_loss=True \
model.peft.peft_scheme=none \
model.data.train_ds.file_names=${TRAIN_DS} \
model.data.validation_ds.file_names=${VALID_DS} \
model.data.test_ds.file_names=${TEST_DS} \
model.data.train_ds.concat_sampling_probabilities=${CONCAT_SAMPLING_PROBS} \
model.data.train_ds.max_seq_length=2048 \
model.data.validation_ds.max_seq_length=2048 \
model.data.train_ds.micro_batch_size=1 \
model.data.train_ds.global_batch_size=128 \
model.data.validation_ds.micro_batch_size=1 \
model.data.validation_ds.global_batch_size=128 \
model.data.test_ds.micro_batch_size=1 \
model.data.test_ds.global_batch_size=256 \
model.data.train_ds.num_workers=0 \
model.data.validation_ds.num_workers=0 \
model.data.test_ds.num_workers=0 \
model.data.validation_ds.metric.name=loss \
model.data.test_ds.metric.name=loss \
exp_manager.create_wandb_logger=False \
exp_manager.explicit_log_dir=/projects/t5c/public/ \
exp_manager.resume_if_exists=True \
exp_manager.resume_ignore_no_checkpoint=True \
exp_manager.create_checkpoint_callback=True \
exp_manager.checkpoint_callback_params.monitor=validation_loss \
exp_manager.checkpoint_callback_params.save_best_model=False \
exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True