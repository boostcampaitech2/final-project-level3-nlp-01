from transformers import BertConfig
from transformers import GPT2Config


class StudencEncoderConfig(BertConfig):
    
    def __init__(
        self,
        _name_or_path="bert-base-multilingual-cased",
        architectures=["BertForMaskedLM"],
        vocab_size=119547,
        hidden_size=768,
        num_hidden_layers=1, # Student Layer Length
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        directionality="bidi",
        model_type="bert",
        pooler_fc_size=768,
        pooler_num_attention_heads=12,
        pooler_num_fc_layers=3,
        pooler_size_per_head=128,
        pooler_type="first_token_transform",
        transformers_version="4.12.5",
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        
        self._name_or_path = _name_or_path,
        self.architectures = architectures
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.directionality = directionality
        self.model_type = model_type
        self.pooler_fc_size = pooler_fc_size
        self.pooler_num_attention_heads = pooler_num_attention_heads
        self.pooler_num_fc_layers = pooler_num_fc_layers
        self.pooler_size_per_head = pooler_size_per_head
        self.pooler_type = pooler_type
        self.transformers_version = transformers_version


class StudentDecoderConfig(GPT2Config):
    
    def __init__(
        self,
        _name_or_path="kykim/gpt3-kor-small_based_on_gpt2",
        architectures=["GPT2Model"],
        vocab_size=42000,
        n_positions=2048,
        n_ctx=2048,
        n_embd=768,
        n_layer=1, # = num_hidden_layers = num_student_layers
        n_head=12,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-05,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=3,
        eos_token_id=3,
        transformers_version="4.12.5",
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        gradient_checkpointing=False,
        decoder_start_token_id=2,
        is_encoder_decoder=True,
        model_type="gpt2",
        **kwargs
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.transformers_version = transformers_version
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn
        self.gradient_checkpointing = gradient_checkpointing
        self.decoder_start_token_id = decoder_start_token_id
        self.architectures = architectures
        self._name_or_path = _name_or_path
        self.is_encoder_decoder = is_encoder_decoder
        self.model_type = model_type