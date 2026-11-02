from pydantic import BaseModel


class RunConfig(BaseModel):
    optimizer_lr_min: float
    optimizer_lr_max: float
    optimizer_warmup_iters: int
    optimizer_cosine_cycle_iters: int
    optimizer_weight_decay: float
    optimizer_beta1: float
    optimizer_beta2: float
    optimizer_eps: float
    optimizer_muon: bool = False  # older runs' config.json predates this field; they used AdamW only
    model_vocab_size: int
    model_context_length: int
    model_num_layers: int
    model_d_model: int
    model_d_ff: int
    model_num_heads: int
    model_eps: float
    model_rope_theta: int | None
    model_tie_embeddings: bool = False  # older runs' config.json predates this field; they were untied
    model_qk_norm: bool = False  # older runs' config.json predates this field; they had no QK-norm
    model_logit_soft_cap: float | None = None  # older runs' config.json predates this field; they had no cap
    gradient_max_l2_norm: float
    gradient_eps: float
    iter: int
    device: str
    batch_size: int
    tokenizer_dir: str
    train_data: str
    val_data: str | None
    val_batches: int
    val_interval: int
    log_interval: int
    runs_dir: str
    run_name: str
    checkpoints_iter: int
    checkpoint_src: str | None
    wandb_project: str | None = None  # older runs' config.json predates this field
    git_commit: str | None = None  # code version the run was trained with; "-dirty" means uncommitted changes
    time_budget_minutes: float | None = None  # with a budget, training ended on time rather than at iter
    time_reserve_minutes: float = 0.0
    z_loss_weight: float = 0.0  # older runs' config.json predates this field; they had no z-loss
