from calendar import TUESDAY
from operator import mod
import torch
from torch import nn
from transformers.models.wav2vec2.configuration_wav2vec2 import Wav2Vec2Config
from transformers.models.wav2vec2.feature_extraction_wav2vec2 import Wav2Vec2FeatureExtractor as Wav2Vec2FeatureExtractorMask
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, \
     Wav2Vec2GumbelVectorQuantizer, _compute_mask_indices, _sample_negative_indices, Wav2Vec2ForPreTrainingOutput
from transformers import Wav2Vec2PreTrainedModel

class TSWav2Vec(Wav2Vec2PreTrainedModel):
    def __init__(self, mode="pre-train"):

        self.mode = mode
        if self.mode == "pre-train":
            mask_time_prob = 0.65
        elif self.mode == "forecasting":
            mask_time_prob = 0
        else:
            assert False, "Wrong mode: {}".format(mode)

        config = Wav2Vec2Config(vocab_size=32, hidden_size=48, num_hidden_layers=4, num_attention_heads=6,
                                intermediate_size=192, hidden_act="gelu",
                                layerdrop=0.1, initializer_range=0.02, layer_norm_eps=1e-5,
                                feat_extract_norm="layer", feat_extract_activation="gelu",
                                conv_dim=(32, 32, 32, 32, 32), conv_stride=(1, 1, 2, 2, 2), conv_kernel=(2, 2, 2, 2, 2),
                                num_conv_pos_embeddings=12, num_conv_pos_embedding_groups=4,
                                conv_bias=True, do_stable_layer_norm=True,
                                apply_spec_augment=True, mask_time_prob=mask_time_prob, mask_time_length=5, mask_time_min_masks=2,
                                num_codevectors_per_group=120, num_codevector_groups=2, contrastive_logits_temperature=0.1,
                                num_negatives=100, codevector_dim=48, proj_codevector_dim=48, diversity_loss_weight=0.1)

        super().__init__(config)
        self.config = config
        self.feature_extractor_mask = Wav2Vec2FeatureExtractorMask.from_pretrained("/data/research/time_series/transformers-4.18.0/preprocessor_config.json")

        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout_features = nn.Dropout(config.feat_quantizer_dropout)

        self.quantizer = Wav2Vec2GumbelVectorQuantizer(config)
        self.post_init()

        # make sure that project_hid & project_q are initialized like normal linear layers
        self.project_hid = nn.Linear(config.hidden_size, config.proj_codevector_dim)
        self.project_q = nn.Linear(config.codevector_dim, config.proj_codevector_dim)

    def set_gumbel_temperature(self, temperature: int):
        self.quantizer.temperature = temperature

    def freeze_feature_extractor(self):
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        self.wav2vec2.feature_extractor._freeze_parameters()
    def compute_contrastive_logits(self, target_features: torch.FloatTensor, negative_features: torch.FloatTensor,
        predicted_features: torch.FloatTensor, temperature: int = 0.1):

        target_features = torch.cat([target_features, negative_features], dim=0)
        logits = torch.cosine_similarity(predicted_features.float(), target_features.float(), dim=-1).type_as(
            target_features
        )
        # apply temperature
        logits = logits / temperature
        return logits

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None, epoch: int = None, **kwargs) -> torch.Tensor:
        # reshape
        history_data = history_data.permute(0, 2, 3, 1)     # B, N, 1, L * P
        batch_size, num_node, channel, length = history_data.size()
        assert channel == 1, "we now only support channel=1"
        input_values = history_data.contiguous().view(batch_size * num_node, length)
        return_dict = True
        output_attentions = False
        output_hidden_states = True
 # =======================reformat=====================================================================
        features = [{"input_values": input_values[i, :]} for i in range(input_values.size()[0])]
        batch = self.feature_extractor_mask.pad(
            features,
            padding="longest",
            pad_to_multiple_of=None,
            return_tensors="pt",
        )
        batch_size = batch["input_values"].shape[0]
        mask_indices_seq_length = self._get_feat_extract_output_lengths(batch["input_values"].shape[-1])
        # make sure masked sequence length is a Python scalar
        mask_indices_seq_length = int(mask_indices_seq_length)

        # make sure that no loss is computed on padded inputs
        if batch.get("attention_mask") is not None:
            # compute real output lengths according to convolution formula
            batch["sub_attention_mask"] = self._get_feature_vector_attention_mask(
                mask_indices_seq_length, batch["attention_mask"]
            )

        features_shape = (batch_size, mask_indices_seq_length)

        # sample randomly masked indices
        mask_time_indices = _compute_mask_indices(
            features_shape,
            self.config.mask_time_prob,
            self.config.mask_time_length,
            attention_mask=batch.get("sub_attention_mask"),
        )

        # sample negative indices
        sampled_negative_indices = _sample_negative_indices(
            features_shape,
            self.config.num_negatives,
            mask_time_indices=mask_time_indices,
        )

        mask_time_indices = torch.from_numpy(mask_time_indices).to(input_values.device)
        sampled_negative_indices = torch.from_numpy(sampled_negative_indices).to(input_values.device)
        attention_mask = batch["attention_mask"]
        # ================================================================================
        if mask_time_indices is not None:
            mask_time_indices = mask_time_indices.to(torch.bool)

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mask_time_indices=mask_time_indices,
            return_dict=return_dict,
        )

        # 1. project all transformed features (including masked) to final vq dim
        transformer_features = self.project_hid(outputs[0])

        # 2. quantize all (unmasked) extracted features and project to final vq dim
        extract_features = self.dropout_features(outputs[1])

        if attention_mask is not None:
            # compute reduced attention_mask correponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        quantized_features, codevector_perplexity = self.quantizer(
            extract_features, mask_time_indices=mask_time_indices
        )
        quantized_features = self.project_q(quantized_features)

        loss = contrastive_loss = diversity_loss = None
        if sampled_negative_indices is not None:
            batch_size, sequence_length, hidden_size = quantized_features.shape

            # for training, we sample negatives
            # 3. sample K negatives (distractors) quantized states for contrastive loss
            # if attention_mask is passed, make sure that padded feature vectors cannot be sampled
            # sample negative quantized vectors BTC => (BxT)C
            negative_quantized_features = quantized_features.view(-1, hidden_size)[
                sampled_negative_indices.long().view(-1)
            ]
            negative_quantized_features = negative_quantized_features.view(
                batch_size, sequence_length, -1, hidden_size
            ).permute(2, 0, 1, 3)

            # 4. compute logits, corresponding to `logs = sim(c_t, [q_t, \sim{q}_t]) / \kappa`
            # of equation (3) in https://arxiv.org/pdf/2006.11477.pdf
            logits = self.compute_contrastive_logits(
                quantized_features[None, :],
                negative_quantized_features,
                transformer_features,
                self.config.contrastive_logits_temperature,
            )

            # 5. if a negative vector is identical to the positive (i.e. when codebook utilization is low),
            # its cosine similarity will be masked
            neg_is_pos = (quantized_features == negative_quantized_features).all(-1)

            if neg_is_pos.any():
                logits[1:][neg_is_pos] = float("-inf")

            # 6. compute contrastive loss \mathbf{L}_m = cross_entropy(logs) =
            # -log(exp(sim(c_t, q_t)/\kappa) / \sum_{\sim{q}} exp(sim(c_t, \sim{q})/\kappa))
            logits = logits.transpose(0, 2).reshape(-1, logits.size(0))
            target = ((1 - mask_time_indices.long()) * -100).transpose(0, 1).flatten()

            contrastive_loss = nn.functional.cross_entropy(logits.float(), target, reduction="sum")
            # 7. compute diversity loss: \mathbf{L}_d
            num_codevectors = self.config.num_codevectors_per_group * self.config.num_codevector_groups
            diversity_loss = ((num_codevectors - codevector_perplexity) / num_codevectors) #* mask_time_indices.sum()
            contrastive_loss = contrastive_loss / mask_time_indices.sum()
            # 8. \mathbf{L} = \mathbf{L}_m + \alpha * \mathbf{L}_d
            loss = contrastive_loss + self.config.diversity_loss_weight * diversity_loss

        if not return_dict:
            if loss is not None:
                return (loss, transformer_features, quantized_features, codevector_perplexity) + outputs[2:]
            return (transformer_features, quantized_features, codevector_perplexity) + outputs[2:]

        return Wav2Vec2ForPreTrainingOutput(
            loss=loss,
            projected_states=transformer_features,
            projected_quantized_states=quantized_features,
            codevector_perplexity=codevector_perplexity,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            contrastive_loss=contrastive_loss,
            diversity_loss=diversity_loss,
        )
